import torch
import torch.nn as nn
import os


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = []

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat
            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]
    

class LightGCN(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(LightGCN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool

        self.alpha = args_config.alpha
        self.tau = args_config.tau
        print(f"This is {os.path.basename(__file__)} working...")


        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None, epoch=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        user_pos = batch['observed_pos_items']      # torch.Size([2048, 30])

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
              
        neg_user_embs=self.negative_sampling(user_gcn_emb, item_gcn_emb, user, neg_item, pos_item, user_pos)
        batch_loss1,mf_loss1,emb_loss1=self.create_bpr_loss(user_gcn_emb[user], item_gcn_emb[pos_item], neg_user_embs)
        self._check_nan(batch_loss1)
        return batch_loss1,mf_loss1,emb_loss1

    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item, user_pos):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """Hard Negative Boundary Definition"""
        n_e = item_gcn_emb[neg_candidates]      # [batch_size, n_negs, n_hops+1, channel]
        scores = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1)       #[batch_size, n_negs, n_hops+1]
        indices_max = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        neg_items_embedding_hardest = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices_max, :]   #   [batch_size, n_hops+1, channel]

        """Historical Interaction-Aware Positive Augmentation"""
        observed_pos_e = item_gcn_emb[user_pos]     # [batch_size, n_pos, n_hops+1, channel]
        observed_pos_score = (s_e.unsqueeze(dim=1) * observed_pos_e).sum(dim=-1)        # [batch_size, n_pos, n_hops+1]
        observed_pos_score = torch.exp(observed_pos_score)      # [batch_size, n_pos, n_hops+1]
              
        pos_score = (s_e * p_e).sum(dim=-1)     # [batch_size, n_hops+1]
        pos_score = self.alpha * torch.exp(pos_score) 
        
        # [batch_size, n_pos, n_hops+1] + [batch_size, n_hops+1]
        total_sum_pp = observed_pos_score.sum(dim=1) + pos_score      # [batch_size, n_hops+1]
        weight1 =  (observed_pos_score / total_sum_pp.unsqueeze(dim=1)).unsqueeze(dim=-1)      # [batch_size, items, channel, 1]
        weight2 = (pos_score / total_sum_pp).unsqueeze(dim=-1)      # [batch_size, channel]
        p_e_ = (weight1 * observed_pos_e).sum(dim=1) + weight2 * p_e        # [batch_size, channel]
        
        """Relevance-Aware Negative Mixup"""
        _pos_score = (s_e * p_e_).sum(dim=-1)       # [batch_size, n_hops+1]
        _pos_score = self.alpha * torch.exp(_pos_score)
        
        neg_score = (s_e * neg_items_embedding_hardest).sum(dim=-1)     # [batch_size, n_hops+1]
        neg_score = torch.exp(neg_score)
        
        total_sum_pn = neg_score + _pos_score      # [batch_size, n_hops+1]
        neg_weight = (neg_score / total_sum_pn).unsqueeze(dim=-1)
        pos_weight = 1 - neg_weight
        n_e_ =  pos_weight * p_e_ + neg_weight * neg_items_embedding_hardest  # mixing
        
        return n_e_

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=-1)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), axis=-1)  # [batch_size, K]
        

        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores)))

        # cul regularizer
        regularize0 = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:,0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * (regularize0) / batch_size

        return mf_loss+emb_loss, mf_loss, emb_loss
