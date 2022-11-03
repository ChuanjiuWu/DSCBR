#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 

from loss1 import DCL
from loss1.dcl import DCLW


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)



    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]

    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class DSCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_positive=conf["positive_t"]

        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph



        # generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()

        ## cal    b-b graph  用以获得bundle间的相似度
        self.bb_graph = self.get_ii_constraint_mat(self.ub_graph, 3)

        self.init_md_dropouts()

        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]
        self.kk=10


    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)


    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)


    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)


    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)


    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)


    def get_ii_constraint_mat(self,train_mat, num_neighbors, ii_diagonal_zero=True):

        print('Computing \\Omega for the item-item graph... ')
        A = train_mat.T.dot(train_mat)  # B * B
        n_items = A.shape[0]
        res_mat = torch.zeros((n_items, num_neighbors))
        res_sim_mat = torch.zeros((n_items, num_neighbors))
        if ii_diagonal_zero:
            A[range(n_items), range(n_items)] = 0

        for i in range(n_items):
            row =   torch.from_numpy(A.getrow(i).toarray()[0])
            row_sims, row_idxs = torch.topk(row, num_neighbors)
            res_mat[i] = row_idxs
            res_sim_mat[i] = row_sims
            if i % 15000 == 0:
                print('i-i constraint matrix {} ok'.format(i))

        print('Computation \\Omega OK!')
        return res_mat.long()


    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test: # !!! important
                features = mess_dropout(features)

            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature


    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature


    def propagate(self, test=False):
        #  =============================  item level propagation  =============================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature, self.items_feature, self.item_level_dropout, test)
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature, self.items_feature, self.item_level_dropout, test)

        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)

        #  ============================= bundle level propagation =============================
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)
        else:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test)

        users_feature = [IL_users_feature, BL_users_feature]
        bundles_feature = [IL_bundles_feature, BL_bundles_feature]

        return users_feature, bundles_feature

    def FN_mask(self,sim):
        # mask(Removing False Negative Samples)
        max_ = torch.max(sim, dim=-1, keepdim=True).values
        min_ = torch.min(sim, dim=-1, keepdim=True).values
        sim_ = (sim - min_) / (max_ - min_)
        eye_matrix=torch.eye(sim.shape[0],dtype=torch.long).to("cuda:0")
        sim_[eye_matrix==1]=0

        return sim_>=1.0

    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)

        pos_score = torch.sum(pos * aug, dim=1)  # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0))  # [batch_size, batch_size]

 # 过滤 false  negative
        # m_1 = self.FN_mask(ttl_score)
        # ttl_score[m_1 == 1] = float("-inf")
#-----------------------------------------------

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]
        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss
    def scl(self,pred):
        # pred: [bs, 1+neg_num]
        if pred.shape[1] > 2:
            negs = pred[:, 1:]
            pos = pred[:, 0]
        else:
            negs = pred[:, 1]
            pos = pred[:, 0]

        # The negative samples are filtered and the similarity is calculated with the anchor, and the lower one is more likely to be true negative
# parts1
#         temp=-negs
#
#         tb = torch.ones(negs.shape)   # create new tensor for 1,0
#         # set `1` to topk indices
#         tb[(torch.arange(len(negs)).unsqueeze(1), torch.topk(temp, self.kk).indices)] = 0
#
#         negs[tb == 1] =float("-inf")


        negs=torch.exp(negs/self.c_temp).sum(dim=-1)

        pos=torch.exp(pos/self.c_temp)
        loss = (- torch.log(pos / (pos + negs))).mean()

        return loss

    def cal_loss(self, users_feature, bundles_feature):
        # IL: item_level, BL: bundle_level
        # [bs, 1, emb_size]
        IL_users_feature, BL_users_feature = users_feature
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_feature

        # cal  cross view cl_loss
        u_cross_view_cl = self.cal_c_loss(IL_users_feature, BL_users_feature)
        b_cross_view_cl = self.cal_c_loss(IL_bundles_feature, BL_bundles_feature)

        c_losses = [u_cross_view_cl, b_cross_view_cl]

        c_loss = sum(c_losses) / len(c_losses)

        return c_loss


    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()
            self.get_bundle_agg_graph()

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_feature, bundles_feature = self.propagate()




        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]
        bundles_embedding = [i[bundles] for i in bundles_feature]

        c_loss = self.cal_loss(users_embedding, bundles_embedding)

        ##  cal  scl
        IL_users_feature, BL_users_feature = users_embedding
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature = bundles_embedding

        pred_scl1 = torch.sum(IL_users_feature * BL_bundles_feature, 2)    # print('pred_ucl1',pred_ucl1.shape)  [2048, 11]
        pred_scl2 = torch.sum(BL_users_feature * IL_bundles_feature, 2)
        scl_loss_ori = self.scl(pred_scl1)
        scl_loss_ori += self.scl(pred_scl2)

        ### 使用uu  bb 图的部分

        pos_bundle = bundles[:, 0]
        bundle_friends = self.bb_graph[pos_bundle].to(self.device)  # [batchsize,k]

    #top1

        scl_loss=0
        for ii in range(self.num_positive):
            bundle_friend0 = bundle_friends[:, ii]
            bundle_friend0 = bundle_friend0.unsqueeze(1)
            new_bundles0 = torch.cat((bundle_friend0, bundles[:, 1:]), dim=1).to(self.device)
            users_embedding_new0 = [i[users].expand(-1, new_bundles0.shape[1], -1) for i in users_feature]
            bundles_embedding_new0 = [i[new_bundles0] for i in bundles_feature]

            IL_users_feature0, BL_users_feature0 = users_embedding_new0  # [bs, 1+neg_num, emb_size]
            IL_bundles_feature0, BL_bundles_feature0 = bundles_embedding_new0

            pred_scl1 = torch.sum(IL_users_feature0 * BL_bundles_feature0, 2)
            # print('pred_ucl1',pred_ucl1.shape)  [2048, 11]
            pred_scl2 = torch.sum(BL_users_feature0 * IL_bundles_feature0, 2)
            scl_loss += self.scl(pred_scl1)
            scl_loss += self.scl(pred_scl2)
    #
    # # top2
    #     bundle_friend1 = bundle_friends[:, 1]
    #     bundle_friend1 = bundle_friend1.unsqueeze(1)
    #     new_bundles1 = torch.cat((bundle_friend1, bundles[:, 1:]), dim=1).to(self.device)
    #
    #     users_embedding_new1 = [i[users].expand(-1, new_bundles1.shape[1], -1) for i in users_feature]
    #     bundles_embedding_new1 = [i[new_bundles1] for i in bundles_feature]
    #     IL_users_feature1, BL_users_feature1 = users_embedding_new1  # [bs, 1+neg_num, emb_size]
    #     IL_bundles_feature1, BL_bundles_feature1 = bundles_embedding_new1
    #     pred_ucl3 = torch.sum(IL_users_feature1 * BL_bundles_feature1, 2)
    #     # print('pred_ucl1',pred_ucl1.shape)  [2048, 11]
    #     pred_ucl4 = torch.sum(BL_users_feature1 * IL_bundles_feature1, 2)
    #     ucl_loss += self.ucl(pred_ucl3)
    #     ucl_loss += self.ucl(pred_ucl4)
    #top3
        # bundle_friend2 = bundle_friends[:, 2]
        # bundle_friend2 = bundle_friend2.unsqueeze(1)
        # new_bundles2 = torch.cat((bundle_friend2, bundles[:, 1:]), dim=1).to(self.device)
        # users_embedding_new2 = [i[users].expand(-1, new_bundles2.shape[1], -1) for i in users_feature]
        # bundles_embedding_new2 = [i[new_bundles2] for i in bundles_feature]
        # IL_users_feature2, BL_users_feature2 = users_embedding_new2  # [bs, 1+neg_num, emb_size]
        # IL_bundles_feature2, BL_bundles_feature2 = bundles_embedding_new2
        # pred_ucl5 = torch.sum(IL_users_feature2 * BL_bundles_feature2, 2)
        # # print('pred_ucl1',pred_ucl1.shape)  [2048, 11]
        # pred_ucl6 = torch.sum(BL_users_feature2 * IL_bundles_feature2, 2)
        # ucl_loss += self.ucl(pred_ucl5)
        # ucl_loss += self.ucl(pred_ucl6)


        # ucl_loss=scl_loss_ori +0.2* scl_loss
        scl_loss=scl_loss_ori+0.2 * scl_loss





        pred = torch.sum(IL_users_feature * IL_bundles_feature, 2) + torch.sum(BL_users_feature * BL_bundles_feature, 2)

        bpr_loss = cal_bpr_loss(pred)


        return bpr_loss, c_loss,scl_loss

    # Dynamic Negative Sampling First, a batch of negative samples are evenly sampled,
    # and then a few of the negative samples are selected to have the lowest similarity scores, or of course, the highest. (The higher the score, the harder it may be)

    def sample_neg(self,bundles,users_embedding,bundles_embedding,sample_num):
        users_embedding=users_embedding[:,1:,:]
        neg_bundles_embedding=bundles_embedding[:,1:,:]

        users_embedding = F.normalize(users_embedding, p=2, dim=2)
        neg_bundles_embedding = F.normalize(neg_bundles_embedding, p=2, dim=2)


        # item_candi=bundles[:,1:]

        sim_score=torch.sum(users_embedding * neg_bundles_embedding,2)
        sim_score = -sim_score
        topk= torch.topk(sim_score,3)



    '''
        Sampling according to probability
    '''
        # item_candi=item_candi.tolist()
        #
        # prob=torch.exp(torch.sum(users_embedding * neg_bundles_embedding, 2)) #(batch_size,neg_num)
        # prob_sum=torch.sum(prob,dim=1).unsqueeze(-1)
        # prob=(prob/prob_sum)**self.ia
        # prob_list=prob.tolist()
        #
        # neg_all = []
        #
        # for i in range(bundles.shape[0]):
        #     random_index_list = np.random.choice(item_candi[i], sample_num, prob_list[i])
        #     neg_all.append(random_index_list)


        # neg_all = torch.tensor(neg_all).to(self.device)
        # item_all=torch.cat([bundles[:,0].unsqueeze(-1),neg_all],1)
        # return item_all

    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature]
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature

        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        return scores
