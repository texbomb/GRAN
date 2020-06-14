import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
EPS = np.finfo(np.float32).eps

__all__ = ['GRANMixtureBernoulli']


class GNN(nn.Module):

  def __init__(self,
               msg_dim,
               node_state_dim,
               edge_feat_dim,
               node_attributes_dim,
               edge_attributes_dim,
               num_prop=1,
               num_layer=1,
               has_attention=True,
               att_hidden_dim=128,
               has_residual=False,
               has_graph_output=False,
               output_hidden_dim=128,
               graph_output_dim=None):
    super(GNN, self).__init__()
    self.msg_dim = msg_dim
    self.node_state_dim = node_state_dim
    self.edge_feat_dim = edge_feat_dim
    self.num_prop = num_prop
    self.num_layer = num_layer
    self.has_attention = has_attention
    self.has_residual = has_residual
    self.att_hidden_dim = att_hidden_dim
    self.has_graph_output = has_graph_output
    self.output_hidden_dim = output_hidden_dim
    self.graph_output_dim = graph_output_dim
    self.node_attribute_dim = node_attributes_dim
    self.edge_attribute_dim = edge_attributes_dim

    self.update_func = nn.ModuleList([
        nn.GRUCell(input_size=self.msg_dim, hidden_size=self.node_state_dim)
        for _ in range(self.num_layer)
    ])

    self.msg_func = nn.ModuleList([
        nn.Sequential(
            *[
                nn.Linear( (self.edge_attribute_dim + 1) * self.node_state_dim + self.edge_feat_dim + self.node_attribute_dim, 
                          self.msg_dim),  
                #nn.Dropout(p=0.2),              
                nn.ReLU(),
                nn.Linear(self.msg_dim, self.msg_dim)
            ]) for _ in range(self.num_layer)
    ])

    if self.has_attention:
      self.att_head = nn.ModuleList([
          nn.Sequential(
              *[
                  nn.Linear((self.edge_attribute_dim + 1) * self.node_state_dim + self.edge_feat_dim + self.node_attribute_dim,
                            self.att_hidden_dim),
                  nn.ReLU(),
                  nn.Linear(self.att_hidden_dim, self.msg_dim),
                  nn.Sigmoid()
              ]) for _ in range(self.num_layer)
      ])

    if self.has_graph_output:
      self.graph_output_head_att = nn.Sequential(*[
          nn.Linear(self.node_state_dim, self.output_hidden_dim),
          nn.ReLU(),
          nn.Linear(self.output_hidden_dim, 1),
          nn.Sigmoid()
      ])

      self.graph_output_head = nn.Sequential(
          *[nn.Linear(self.node_state_dim, self.graph_output_dim)])

  def _prop(self, state, edge, edge_feat, node_attributes, edge_attributes, layer_idx=0):
    ### compute message
    state_diff = state[edge[:, 0], :] - state[edge[:, 1], :]
    if self.edge_feat_dim > 0:
      edge_input = torch.cat([state_diff, edge_feat], dim=1)
    else:
      edge_input = state_diff

    if self.edge_attribute_dim > 0:
      for i in range(len(edge_attributes)):
        attribute_diff = edge_attributes[i,edge[:, 0], :] - edge_attributes[i,edge[:, 1], :]
        edge_input = torch.cat([edge_input, attribute_diff], dim=1)

    if self.node_attribute_dim > 0:  
      for i in range(len(node_attributes)):
        attribute_diff = node_attributes[i,edge[:, 0]] - node_attributes[i,edge[:, 1]]
        edge_input = torch.cat([edge_input, attribute_diff.reshape(-1,1)], dim=1)

    msg = self.msg_func[layer_idx](edge_input)    

    ## attention on messages
    if self.has_attention:
      att_weight = self.att_head[layer_idx](edge_input)
      msg = msg * att_weight

    ### aggregate message by sum
    state_msg = torch.zeros(state.shape[0], msg.shape[1]).to(state.device)
    scatter_idx = edge[:, [1]].expand(-1, msg.shape[1])
    state_msg = state_msg.scatter_add(0, scatter_idx, msg)

    ### state update
    state = self.update_func[layer_idx](state_msg, state)
    return state

  def forward(self, node_feat, edge, edge_feat, node_attributes, edge_attributes, graph_idx=None):
    """
      N.B.: merge a batch of graphs as a single graph

      node_feat: N X D, node feature
      edge: M X 2, edge indices
      edge_feat: M X D', edge feature
      graph_idx: N X 1, graph indices
    """

    state = node_feat
    prev_state = state
    for ii in range(self.num_layer):
      if ii > 0:
        state = F.relu(state)

      for jj in range(self.num_prop):
        state = self._prop(state, edge, edge_feat=edge_feat, node_attributes=node_attributes, edge_attributes=edge_attributes, layer_idx=ii)

    if self.has_residual:
      state = state + prev_state

    if self.has_graph_output:
      num_graph = graph_idx.max() + 1
      node_att_weight = self.graph_output_head_att(state)
      node_output = self.graph_output_head(state)

      # weighted average
      reduce_output = torch.zeros(num_graph,
                                  node_output.shape[1]).to(node_feat.device)
      reduce_output = reduce_output.scatter_add(0,
                                                graph_idx.unsqueeze(1).expand(
                                                    -1, node_output.shape[1]),
                                                node_output * node_att_weight)

      const = torch.zeros(num_graph).to(node_feat.device)
      const = const.scatter_add(
          0, graph_idx, torch.ones(node_output.shape[0]).to(node_feat.device))

      reduce_output = reduce_output / const.view(-1, 1)

      return reduce_output
    else:
      return state


class GRANMixtureBernoulli(nn.Module):
  """ Graph Recurrent Attention Networks """

  def __init__(self, config):
    super(GRANMixtureBernoulli, self).__init__()
    self.config = config
    self.device = config.device
    self.max_num_nodes = config.model.max_num_nodes
    self.hidden_dim = config.model.hidden_dim
    self.is_sym = config.model.is_sym
    self.block_size = config.model.block_size
    self.sample_stride = config.model.sample_stride
    self.num_GNN_prop = config.model.num_GNN_prop
    self.num_GNN_layers = config.model.num_GNN_layers
    self.edge_weight = config.model.edge_weight if hasattr(
        config.model, 'edge_weight') else 1.0
    self.dimension_reduce = config.model.dimension_reduce
    self.has_attention = config.model.has_attention
    self.num_canonical_order = config.model.num_canonical_order
    self.output_dim = 1
    self.num_mix_component = config.model.num_mix_component
    self.has_rand_feat = True # use random feature instead of 1-of-K encoding
    self.att_edge_dim = 64
    self.node_attributes_dim = sum([self.config.attributes[i]['node_feature'] for i in self.config.attributes])
    self.edge_attributes_dim = len(self.config.attributes) -self.node_attributes_dim 
    self.embedding_dim = config.model.embedding_dim

    self.output_theta = nn.Sequential(
        nn.Linear(self.embedding_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.output_dim * self.num_mix_component))
    
    self.output_node_attributes = nn.ModuleList([
      nn.Sequential(
        *[
          nn.Linear(self.embedding_dim, self.hidden_dim),
          #nn.BatchNorm1d(self.hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          #nn.BatchNorm1d(self.hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          #nn.BatchNorm1d(self.hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.hidden_dim, self.output_dim * self.num_mix_component)
        ]) for _ in range(self.node_attributes_dim)
    ])

    self.output_edge_attributes = nn.ModuleList([
      nn.Sequential(
        *[
          nn.Linear(self.embedding_dim, self.hidden_dim),
          #nn.BatchNorm1d(self.hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          #nn.BatchNorm1d(self.hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.hidden_dim, self.hidden_dim),
          #nn.BatchNorm1d(self.hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.hidden_dim, 1)
        ]) for _ in range(self.edge_attributes_dim)
    ])

    self.output_alpha = nn.Sequential(
        nn.Linear(self.embedding_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(self.hidden_dim, self.num_mix_component),
        nn.LogSoftmax(dim=1))

    if self.dimension_reduce:
      self.embedding_dim = config.model.embedding_dim
      self.decoder_input = nn.Sequential(
          nn.Linear(self.max_num_nodes, self.embedding_dim))
      self.decoder_pos_input = nn.Sequential(
          nn.Linear(self.max_num_nodes, self.embedding_dim))
    else:
      self.embedding_dim = self.max_num_nodes

    self.decoder = GNN(
        msg_dim=self.hidden_dim,
        node_state_dim=self.embedding_dim,
        edge_feat_dim=2 * self.att_edge_dim,
        node_attributes_dim= self.node_attributes_dim,
        edge_attributes_dim= self.edge_attributes_dim,
        num_prop=self.num_GNN_prop,
        num_layer=self.num_GNN_layers,
        has_attention=self.has_attention)

    ### Loss functions
    pos_weight = torch.ones([1]) * self.edge_weight
    self.adj_loss_func = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight, reduction='none')
        
    self.node_attribute_loss_func = nn.MSELoss(reduction='none')
    self.edge_attribute_loss_func = nn.MSELoss(reduction='none')
  def _inference(self,
                 A_pad=None,
                 edges=None,
                 node_idx_gnn=None,
                 node_idx_feat=None,
                 att_idx=None,
                 node_attributes=None,
                 edge_attributes=None):
    """ generate adj in row-wise auto-regressive fashion """

    B, C, N_max, _ = A_pad.shape
    H = self.hidden_dim
    K = self.block_size
    A_pad = A_pad.view(B * C * N_max, -1)
    for attribute in edge_attributes:
      edge_attributes[attribute] =  edge_attributes[attribute].view(B * C * N_max, -1)  

    if self.dimension_reduce:
      node_feat = self.decoder_input(A_pad)  # BCN_max X H
      # //Johan decoding edge attributes, no need for node attributes, as they are only 1-dimensional
      for attribute in edge_attributes:
        edge_attributes[attribute] = self.decoder_input(edge_attributes[attribute]) 


    else:
      node_feat = A_pad  # BCN_max X N_max


    ### GNN inference
    # pad zero as node feature for newly generated nodes (1st row)
    node_feat = F.pad(
        node_feat, (0, 0, 1, 0), 'constant', value=0.0)  # (BCN_max + 1) X N_max
    for attribute in node_attributes:
       node_attributes[attribute] = F.pad(
            node_attributes[attribute], (1, 0), 'constant', value=0.0)
    for attribute in edge_attributes:
        edge_attributes[attribute] = F.pad(
            edge_attributes[attribute], (0, 0, 1, 0), 'constant', value=0.0)  # (BCN_max + 1) X N_max


    # create symmetry-breaking edge feature for the newly generated nodes
    att_idx = att_idx.view(-1, 1)

    if self.has_rand_feat:
      # create random feature
      att_edge_feat = torch.zeros(edges.shape[0],
                                  2 * self.att_edge_dim).to(node_feat.device)
      idx_new_node = (att_idx[[edges[:, 0]]] >
                      0).long() + (att_idx[[edges[:, 1]]] > 0).long()
      idx_new_node = idx_new_node.byte().squeeze()
      att_edge_feat[idx_new_node, :] = torch.randn(
          idx_new_node.long().sum(),
          att_edge_feat.shape[1]).to(node_feat.device)
    else:
      # create one-hot feature
      att_edge_feat = torch.zeros(edges.shape[0],
                                  2 * self.att_edge_dim).to(node_feat.device)
      # scatter with empty index seems to cause problem on CPU but not on GPU
      att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
      att_edge_feat = att_edge_feat.scatter(
          1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

    # //johan Prepare attributes selection
    for attribute in node_attributes:
       node_attributes[attribute] = node_attributes[attribute][node_idx_feat]
    for attribute in edge_attributes:
        edge_attributes[attribute] = edge_attributes[attribute][node_idx_feat]
    if node_attributes:
      node_attributes = torch.stack([node_attributes[i] for i in node_attributes]) # B, Ordering, value
    if edge_attributes:
      edge_attributes = torch.stack([edge_attributes[i] for i in edge_attributes]) # B, Ordering, node, value

    # GNN inference
    # N.B.: node_feat is shared by multiple subgraphs within the same batch
    node_state = self.decoder(
        node_feat[node_idx_feat], edges, edge_feat=att_edge_feat, node_attributes=node_attributes, edge_attributes=edge_attributes)

    ### Pairwise predict edges
    diff = node_state[node_idx_gnn[:, 0], :] - node_state[node_idx_gnn[:, 1], :]
    
    
    log_theta = self.output_theta(diff)  # B X (tt+K)K
    log_alpha = self.output_alpha(diff)  # B X (tt+K)K
    log_theta = log_theta.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K
    log_alpha = log_alpha.view(-1, self.num_mix_component)  # B X CN(N-1)/2 X K

    node_attributes_pred = []
    edge_attributes_pred = []
    # Predict node attributes
    for attribute_layer in range(len(node_attributes)):
      node_attributes_pred.append( self.output_node_attributes[attribute_layer](node_state) )
    # Predict edge attributes
    for attribute_layer in range(len(edge_attributes)):
      edge_attributes_pred.append( self.output_edge_attributes[attribute_layer](diff) ) #Predict på edges !!"!" KIG her
    #print(pos)


    return log_theta, log_alpha, node_attributes_pred, edge_attributes_pred


  def _sampling(self, B):
    """ generate adj in row-wise auto-regressive fashion """

    K = self.block_size
    S = self.sample_stride
    H = self.hidden_dim
    N = self.max_num_nodes
    mod_val = (N - K) % S
    if mod_val > 0:
      N_pad = N - K - mod_val + int(np.ceil((K + mod_val) / S)) * S
    else:
      N_pad = N

    node_attributes_dim = 0
    edge_attributes_dim = 0
    for attribute in self.config.attributes:
      if eval(''.join(['self.config.attributes.', attribute, '.node_feature'])):
        node_attributes_dim += 1
      else:
        edge_attributes_dim += 1

    A = torch.zeros(B, N_pad, N_pad).to(self.device)
    edge_A = torch.zeros(B, edge_attributes_dim, N_pad, N_pad).to(self.device)
    node_A = torch.zeros(B, node_attributes_dim, N_pad).to(self.device)

    dim_input = self.embedding_dim if self.dimension_reduce else self.max_num_nodes
    ### cache node state for speed up
    node_state = torch.zeros(B, N_pad, dim_input).to(self.device)
    edge_state = torch.zeros(B, edge_attributes_dim, N_pad, dim_input).to(self.device)
    node_attribute_state = torch.zeros(B, node_attributes_dim, N_pad).to(self.device)



    for ii in range(0, N_pad, S):
      # for ii in range(0, 3530, S):
      jj = ii + K
      if jj > N_pad:
        break

      # reset to discard overlap generation
      A[:, ii:, :] = .0
      A = torch.tril(A, diagonal=-1)
      
      edge_A[:,:, ii:, :] = .0
      edge_A = torch.tril(edge_A, diagonal=-1)

      if ii >= K:
        if self.dimension_reduce:
          node_state[:, ii - K:ii, :] = self.decoder_input(A[:, ii - K:ii, :N])
          edge_state[:,:, ii - K:ii, :] = self.decoder_input(edge_A[:,:, ii - K:ii, :N])

        else:
          node_state[:, ii - K:ii, :] = A[:, ii - S:ii, :N]
          edge_state[:,:, ii - K:ii, :] = edge_A[:,:, ii - S:ii, :N]
        node_attribute_state[:,:, ii - K:ii] = node_A[:,:, ii - S:ii]
      else:
        if self.dimension_reduce:
          node_state[:, :ii, :] = self.decoder_input(A[:, :ii, :N])
          edge_state[:,:, :ii, :] = self.decoder_input(edge_A[:,:, :ii, :N])
        else:
          node_state[:, :ii, :] = A[:, ii - S:ii, :N]
          edge_state[:,:, :ii, :] = edge_A[:,:, ii - S:ii, :N]
        node_attribute_state[:,:,:ii] = node_A[:,:,:ii]
        

      node_state_in = F.pad(
          node_state[:, :ii, :], (0, 0, 0, K), 'constant', value=.0)
      edge_state_in = F.pad(
          edge_state[:,:, :ii, :], (0, 0, 0, K), 'constant', value=.0)
      node_attribute_state_in = F.pad(
          node_attribute_state[:,:, :ii], ( 0, K), 'constant', value=.0)

      ### GNN propagation
      adj = F.pad(
          A[:, :ii, :ii], (0, K, 0, K), 'constant', value=1.0)  # B X jj X jj
      adj = torch.tril(adj, diagonal=-1)
      adj = adj + adj.transpose(1, 2)
      edges = [
          adj[bb].to_sparse().coalesce().indices() + bb * adj.shape[1]
          for bb in range(B)
      ]
      edges = torch.cat(edges, dim=1).t()


      att_idx = torch.cat([torch.zeros(ii).long(),
                           torch.arange(1, K + 1)]).to(self.device)
      att_idx = att_idx.view(1, -1).expand(B, -1).contiguous().view(-1, 1)

      if self.has_rand_feat:
        # create random feature
        att_edge_feat = torch.zeros(edges.shape[0],
                                    2 * self.att_edge_dim).to(self.device)
        idx_new_node = (att_idx[[edges[:, 0]]] >
                        0).long() + (att_idx[[edges[:, 1]]] > 0).long()
        idx_new_node = idx_new_node.byte().squeeze()
        att_edge_feat[idx_new_node, :] = torch.randn(
            idx_new_node.long().sum(), att_edge_feat.shape[1]).to(self.device)
      else:
        # create one-hot feature
        att_edge_feat = torch.zeros(edges.shape[0],
                                    2 * self.att_edge_dim).to(self.device)
        att_edge_feat = att_edge_feat.scatter(1, att_idx[[edges[:, 0]]], 1)
        att_edge_feat = att_edge_feat.scatter(
            1, att_idx[[edges[:, 1]]] + self.att_edge_dim, 1)

      node_state_out = self.decoder(
          node_state_in.view(-1, H), edges,
           edge_feat=att_edge_feat,
           edge_attributes = edge_state_in.view(edge_attributes_dim,-1,H) if edge_attributes_dim else [],
           node_attributes= node_attribute_state_in.view(node_attributes_dim,-1) if node_attributes_dim else [])
      node_state_out = node_state_out.view(B, jj, -1)

      idx_row, idx_col = np.meshgrid(np.arange(ii, jj), np.arange(jj))
      idx_row = torch.from_numpy(idx_row.reshape(-1)).long().to(self.device)
      idx_col = torch.from_numpy(idx_col.reshape(-1)).long().to(self.device)

      diff = node_state_out[:,idx_row, :] - node_state_out[:,idx_col, :]  # B X (ii+K)K X H
      diff = diff.view(-1, node_state.shape[2])
      log_theta = self.output_theta(diff)
      log_alpha = self.output_alpha(diff)

      log_theta = log_theta.view(B, -1, K, self.num_mix_component)  # B X K X (ii+K) X L
      log_theta = log_theta.transpose(1, 2)  # B X (ii+K) X K X L

      log_alpha = log_alpha.view(B, -1, self.num_mix_component)  # B X K X (ii+K)
      prob_alpha = log_alpha.mean(dim=1).exp()      
      alpha = torch.multinomial(prob_alpha, 1).squeeze(dim=1).long()


      prob = []
      for bb in range(B):
        prob += [torch.sigmoid(log_theta[bb, :, :, alpha[bb]])]

        for attribute_layer in range(node_attributes_dim):
          pred = self.output_node_attributes[attribute_layer](node_state_out).reshape(B,-1,self.num_mix_component)
          node_A[:,attribute_layer,ii:jj] = pred[bb,ii:jj,alpha[bb]]
        
        for attribute_layer in range(edge_attributes_dim):
          pred = self.output_edge_attributes[attribute_layer](diff).reshape(B,-1,self.num_mix_component)
          edge_A[:,attribute_layer, ii:jj, :jj] = pred[:,:,alpha[B]]

      prob = torch.stack(prob, dim=0)
      A[:, ii:jj, :jj] = torch.bernoulli(prob[:, :jj - ii, :])

    ### make it symmetric
    if self.is_sym:
      A = torch.tril(A, diagonal=-1)
      A = A + A.transpose(1, 2)

      edge_A = torch.tril(edge_A, diagonal=-1)
      edge_A = edge_A + edge_A.transpose(2, 3)

    #print(node_pos)
    #print(A)

    return A, node_A, edge_A

  def forward(self, input_dict):
    """
      B: batch size
      N: number of rows/columns in mini-batch
      N_max: number of max number of rows/columns
      M: number of augmented edges in mini-batch
      H: input dimension of GNN 
      K: block size
      E: number of edges in mini-batch
      S: stride
      C: number of canonical orderings
      D: number of mixture Bernoulli

      Args:
        A_pad: B X C X N_max X N_max, padded adjacency matrix         
        node_idx_gnn: M X 2, node indices of augmented edges
        node_idx_feat: N X 1, node indices of subgraphs for indexing from feature
                      (0 indicates indexing from 0-th row of feature which is 
                        always zero and corresponds to newly generated nodes)  
        att_idx: N X 1, one-hot encoding of newly generated nodes
                      (0 indicates existing nodes, 1-D indicates new nodes in
                        the to-be-generated block)
        subgraph_idx: E X 1, indices corresponding to augmented edges
                      (representing which subgraph in mini-batch the augmented 
                      edge belongs to)
        edges: E X 2, edge as [incoming node index, outgoing node index]
        label: E X 1, binary label of augmented edges        
        num_nodes_pmf: N_max, empirical probability mass function of number of nodes

      Returns:
        loss                        if training
        list of adjacency matrices  else
    """
    is_sampling = input_dict[
        'is_sampling'] if 'is_sampling' in input_dict else False
    batch_size = input_dict[
        'batch_size'] if 'batch_size' in input_dict else None
    A_pad = input_dict['adj'] if 'adj' in input_dict else None
    node_idx_gnn = input_dict[
        'node_idx_gnn'] if 'node_idx_gnn' in input_dict else None
    node_idx_feat = input_dict[
        'node_idx_feat'] if 'node_idx_feat' in input_dict else None
    att_idx = input_dict['att_idx'] if 'att_idx' in input_dict else None    
    subgraph_idx = input_dict[
        'subgraph_idx'] if 'subgraph_idx' in input_dict else None
    edges = input_dict['edges'] if 'edges' in input_dict else None
    label = input_dict['label'] if 'label' in input_dict else None
    num_nodes_pmf = input_dict[
        'num_nodes_pmf'] if 'num_nodes_pmf' in input_dict else None
    node_attributes_array = input_dict['node_attributes_array'] if 'node_attributes_array' in input_dict else None
    node_attributes_truth = input_dict['node_attributes_truth'] if 'node_attributes_truth' in input_dict else None
    edge_attributes_adj = input_dict['edge_attributes_adj'] if 'edge_attributes_adj' in input_dict else None
    edge_attributes_truth = input_dict['edge_attributes_truth'] if 'edge_attributes_truth' in input_dict else None

    
    N_max = self.max_num_nodes

    if not is_sampling:
      B, _, N, _ = A_pad.shape

      ### compute adj loss
      log_theta, log_alpha, node_attributes_pred, edge_attributes_pred = self._inference(
          A_pad=A_pad,
          edges=edges,
          node_idx_gnn=node_idx_gnn,
          node_idx_feat=node_idx_feat,
          att_idx=att_idx,
          node_attributes=node_attributes_array,
          edge_attributes=edge_attributes_adj)

      num_edges = log_theta.shape[0]

      adj_loss = mixture_bernoulli_loss(label, log_theta, log_alpha,
                                        self.adj_loss_func, subgraph_idx)
      adj_loss = adj_loss * float(self.num_canonical_order)

      total_loss = one_dimensional_loss(node_attributes_pred, node_attributes_truth, self.node_attribute_loss_func, log_alpha, log_theta, self.adj_loss_func, subgraph_idx, label, selection=node_idx_feat)
      edge_attribute_loss = edge_classification(edge_attributes_pred, edge_attributes_truth, self.edge_attribute_loss_func)
      #total_loss = total_loss_function(adj_loss, node_attribute_loss, edge_attribute_loss) 

      return total_loss
      
    else:
      A, node_A, edge_A = self._sampling(batch_size)

      ### sample numbatber of nodes
      num_nodes_pmf = torch.from_numpy(num_nodes_pmf).to(self.device)
      num_nodes = torch.multinomial(
          num_nodes_pmf, batch_size, replacement=True) + 1  # shape B X 1

      A_list = [
          A[ii, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
      ]
      node_A_list = [
          node_A[ii,:, :num_nodes[ii]] for ii in range(batch_size)
      ]
      edge_A_list = [
          edge_A[ii,:, :num_nodes[ii], :num_nodes[ii]] for ii in range(batch_size)
      ]
      #print(A_list)
      return A_list, node_A_list, edge_A_list

# Total loss -> combined adj and positional loss. Need to be tuned with an alpha 

def total_loss_function(adj_loss, *losses):

  losses = list(losses)
  [losses[0].update(losses[i]) for i in range(1, len(losses))]
  # print(f"pos_loss: {pos_loss * 0.15 }")
  # print(f"adj_loss: {adj_loss}")

  total_loss =  adj_loss + sum(losses[0].values())
  #print(adj_loss, pos_loss)
  return total_loss, adj_loss, losses[0]

def edge_classification(pred, truth, pos_loss_func):
  """
    Args:
      pos_true: N X 2, Ground truth positional values
      pos_pred: N X 2, Predicted positional values 
      pos_loss_func: MSE loss
      subgraph_idx: E X 1, see comments above

    Returns:
      loss: mean squared error 
  """
  pred = [pred[i].t() for i in range(len(pred))]
  loss = dict()
  for (x,y) in zip(range(len(pred)),truth):
    loss[y] = torch.sqrt(pos_loss_func(pred[x],truth[y]))
  return loss

def one_dimensional_loss(pred, truth, pos_loss_func, log_alpha, log_theta, adj_loss_func, subgraph_idx, label, selection):
  """
    Args:
      pos_true: N X 2, Ground truth positional values
      pos_pred: N X 2, Predicted positional values 
      pos_loss_func: MSE loss
      subgraph_idx: E X 1, see comments above

    Returns:
      loss: mean squared error 
  """
  num_subgraph = subgraph_idx.max() + 1
  N, K = pred[0].shape
  loss = dict()

    
  const = torch.zeros(num_subgraph).to(pred[0].device)
  const = const.scatter_add(0, subgraph_idx,
                            torch.ones_like(subgraph_idx).float())

  t0 = truth['x'].expand(K,-1).T
  t1 = truth['y'].expand(K,-1).T
  l0 = pos_loss_func(pred[0], t0)
  l1 = pos_loss_func(pred[1], t1)


  x1 = pred[0][label==1]
  x2 = pred[0][selection==0][subgraph_idx[label==1]]
  y1 = pred[1][label==1]
  y2 = pred[1][selection==0][subgraph_idx[label==1]]
  pred_length = torch.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ) 
  pred_bearing = torch.atan((y2-y1)/(x2-x1))

  x1 = truth['x'][label==1]
  x2 = truth['x'][selection==0][subgraph_idx[label==1]]
  y1 = truth['y'][label==1]
  y2 = truth['y'][selection==0][subgraph_idx[label==1]]
  true_length = torch.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ) 
  true_bearing = torch.atan((y2-y1)/(x2-x1))

  length_loss = pos_loss_func(pred_length, true_length.expand(K,-1).T)

  reduce_length = torch.zeros(num_subgraph, K).to(label.device)
  reduce_length = reduce_length.scatter_add(
      0, subgraph_idx[label==1].unsqueeze(1).expand(-1, K), length_loss)  

  bearing_loss = pos_loss_func(pred_bearing, true_bearing.expand(K,-1).T)

  reduce_bearing = torch.zeros(num_subgraph, K).to(label.device)
  reduce_bearing = reduce_bearing.scatter_add(
      0, subgraph_idx[label==1].unsqueeze(1).expand(-1, K), bearing_loss)  


  reduce_l0 = torch.zeros(num_subgraph, K).to(label.device)
  reduce_l0 = reduce_l0.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), l0)
  reduce_l1 = torch.zeros(num_subgraph, K).to(label.device)
  reduce_l1 = reduce_l1.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), l1)
  # reduce_l0 = torch.zeros(num_subgraph, K).to(pred[0].device)
  # reduce_l0 = reduce_l0.scatter_add(
  #     0, subgraph_idx.unsqueeze(1).expand(-1, K), l0)
  # reduce_l0 = reduce_l0 / const.view(-1, 1)

  # reduce_l1 = torch.zeros(num_subgraph, K).to(pred[0].device)
  # reduce_l1 = reduce_l1.scatter_add(
  #     0, subgraph_idx.unsqueeze(1).expand(-1, K), l1)
  # reduce_l1 = reduce_l1 / const.view(-1, 1)


  #Calculate adj and log alpha
  adj_loss = torch.stack(
      [adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1)

  reduce_adj_loss = torch.zeros(num_subgraph, K).to(label.device)
  reduce_adj_loss = reduce_adj_loss.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_loss)

  reduce_log_alpha = torch.zeros(num_subgraph, K).to(pred[0].device)
  reduce_log_alpha = reduce_log_alpha.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha)
  reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)
  reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1)

  #Calculate loss, where alpha is optimized
  log_prob = -reduce_adj_loss - 50*reduce_l0 - 50*reduce_l1 + reduce_log_alpha - 100 * reduce_length - 10 * reduce_bearing
  log_prob = torch.logsumexp(log_prob, dim=1)
  prob_loss = -log_prob.sum() / float(pred[0].shape[0])


  loss['x']= -torch.logsumexp(-50*reduce_l0 + reduce_log_alpha, dim=1).sum() / float(pred[0].shape[0])
  loss['y'] = -torch.logsumexp(-50*reduce_l1 + reduce_log_alpha, dim=1).sum() / float(pred[0].shape[0])
  loss['length'] = -torch.logsumexp(-100 * reduce_length + reduce_log_alpha, dim=1).sum() / float(pred[0].shape[0])
  loss['bearing'] = -torch.logsumexp(-10 * reduce_bearing + reduce_log_alpha, dim=1).sum() / float(pred[0].shape[0])
  adj_loss = -torch.logsumexp(-reduce_adj_loss + reduce_log_alpha, dim=1).sum() / float(pred[0].shape[0])

  total_loss = prob_loss 
  # for (x,y) in zip(range(len(pred)),truth):
  #   loss[y] = pos_loss_func(pred[x],truth[y][selection==0])
  return total_loss, adj_loss, loss

  
def mixture_bernoulli_loss(label, log_theta, log_alpha, adj_loss_func,
                           subgraph_idx):
  """
    Compute likelihood for mixture of Bernoulli model

    Args:
      label: E X 1, see comments above
      log_theta: E X D, see comments above
      log_alpha: E X D, see comments above
      adj_loss_func: BCE loss
      subgraph_idx: E X 1, see comments above

    Returns:
      loss: negative log likelihood
  """

  num_subgraph = subgraph_idx.max() + 1
  K = log_theta.shape[1]
  adj_loss = torch.stack(
      [adj_loss_func(log_theta[:, kk], label) for kk in range(K)], dim=1)

  const = torch.zeros(num_subgraph).to(label.device)
  const = const.scatter_add(0, subgraph_idx,
                            torch.ones_like(subgraph_idx).float())

  reduce_adj_loss = torch.zeros(num_subgraph, K).to(label.device)
  reduce_adj_loss = reduce_adj_loss.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), adj_loss)

  reduce_log_alpha = torch.zeros(num_subgraph, K).to(label.device)
  reduce_log_alpha = reduce_log_alpha.scatter_add(
      0, subgraph_idx.unsqueeze(1).expand(-1, K), log_alpha)
  reduce_log_alpha = reduce_log_alpha / const.view(-1, 1)
  reduce_log_alpha = F.log_softmax(reduce_log_alpha, -1)

  log_prob = -reduce_adj_loss + reduce_log_alpha
  log_prob = torch.logsumexp(log_prob, dim=1)
  loss = -log_prob.sum() / float(log_theta.shape[0])

  # print(log_theta, log_alpha)

  return loss