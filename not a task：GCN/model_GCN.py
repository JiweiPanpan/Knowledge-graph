"""
Laboratory Artificial Intelligence: Deep Learning Lab
Created on Jun 7, 2023
@Team: 02 Jiwei Pan, Ziming Fang
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
from config import device
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, remove_self_loops
import config
from transformers import BertModel, BertTokenizer
import numpy as np


class KGE(nn.Module):
    def __init__(self, model_name, nentity, nrelation, nword, input_dim, hidden_dim, output_dim, dimension):
        super(KGE, self).__init__()
        # Define Hyperparameters
        self.dimension = dimension
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.nword = nword

        # embeddings: Assign one vector to each of the entities and relations, the vectors will be randomly initialized and then trained
        self.entity_embedding = nn.Embedding(self.nentity, self.dimension)
        self.relation_embedding = nn.Embedding(self.nrelation, self.dimension)
        self.word_embedding = nn.Embedding(self.nword, self.dimension)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        # bias
        self.e_bias = nn.Embedding(self.nentity, 1)
        self.r_bias = nn.Embedding(self.nrelation, 1)
        self.w_bias = nn.Embedding(self.nword, 1)

        self.margin = config.MARGIN
        self.margin_neg = config.MARGIN_NEG

        self.init_size = 0.001

        self.initialization()

    def initialization(self):
        self.entity_embedding.weight.data = self.init_size * torch.randn((self.nentity, self.dimension))
        self.relation_embedding.weight.data = self.init_size * torch.randn((self.nrelation, self.dimension))
        self.word_embedding.weight.data = self.init_size * torch.randn((self.nword, self.dimension))
        self.e_bias.weight.data = self.init_size * torch.randn((self.nentity, 1))
        self.r_bias.weight.data = self.init_size * torch.randn((self.nrelation, 1))
        self.w_bias.weight.data = self.init_size * torch.randn((self.nword, 1))

    def forward(self, head, relation, tail):
        pass

    def marginrankingloss(self, positive_score, negative_score, margin):
        rl = nn.ReLU().cuda()
        # print("11111")
        l = torch.sum(rl(negative_score - positive_score + margin))
        return l


# class GCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#
#         # self.conv1 = GCNConv(input_dim, hidden_dim)
#         # self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         # self.conv3 = GCNConv(hidden_dim, output_dim)
#
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = f.relu(x)
#         x = f.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = f.relu(x)
#         x = self.conv3(x, edge_index)
#
#         return x
# # 步骤5：创建并使用GCN模型
#
# class HeadModel(torch.nn.Module):
#     def __init__(self, gcn_model):
#         super(HeadModel, self).__init__()
#         self.gcn_model = gcn_model
#
#     def forward(self, x, edge_index):
#         gcn_output = self.gcn_model(x, edge_index)
#
#
#         return gcn_output
#
#
# class TailModel(torch.nn.Module):
#     def __init__(self, gcn_model):
#         super(TailModel, self).__init__()
#         self.gcn_model = gcn_model
#
#     def forward(self, x, edge_index):
#         gcn_output = self.gcn_model(x, edge_index)
#
#
#         return gcn_output
#
# class RelationModel(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(RelationModel, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.classifier = nn.Linear(self.input_dim, self.output_dim)
#
#     def forward(self, Relation):
#         Relation = self.classifier(Relation)
#         return Relation



class GCNTransE(KGE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""
    '''
    kge_model = TransH(head, relation, tail, ent_num, rel_num, word_num)
    head, relation, tail: tensor variable
    ent_num, rel_num, word_num: int variable, get from KGTrain(), dict size of the dataset


    '''

    def __init__(self, head, relation, tail, num_ent, num_rel, num_word, input_dim, hidden_dim, output_dim, dimension):
        KGE.__init__(self, GCNTransE, num_ent, num_rel, num_word, input_dim, hidden_dim, output_dim, dimension)


        self.head = head
        self.tail = tail
        self.relation = relation
        self.relation_projection = nn.Embedding(self.nrelation, self.dimension)

    ####
    # function computing score
    # @input: head, relation, tail : input tensor variable
    #         head_w, rel_w, tail_w input [list] of words of each entities and relations
    # @output: score , type: tensor
    ####

    def score(self, head, relation, tail, head_w, rel_w, tail_w, edge_index, flag):
        # TransE aims to hold h + r - t = 0, so the distance is ||h + r - t||
        head_e = self.entity_embedding(head)
        rel_e = self.relation_embedding(relation)
        tail_e = self.entity_embedding(tail)
        # bias
        h_bias = self.e_bias(head).squeeze(1)
        t_bias = self.e_bias(tail).squeeze(1)
        r_bias = self.r_bias(relation).squeeze(1)



        # initialize word embeddings containing each line of input triples
        headword_e_combine = None
        relationword_e_combine = None
        tailword_e_combine = None

        for i in range(len(head_w)):
            # transfer word list of entities to tensor
            headword_t = torch.tensor(head_w[i])
            relationword_t = torch.tensor(rel_w[i])
            tailword_t = torch.tensor(tail_w[i])

            # embed words line by line since different dim (different word length of each entity)
            headword_e = self.word_embedding(headword_t)
            relationword_e = self.word_embedding(relationword_t)
            tailword_e = self.word_embedding(tailword_t)

            # shrink dimension from num_word x dimension to 1 x dimension for each entity(i.e. each line)
            headword_e = torch.mean(headword_e, dim=0).unsqueeze(0)
            tailword_e = torch.mean(tailword_e, dim=0).unsqueeze(0)
            relationword_e = torch.mean(relationword_e, dim=0).unsqueeze(0)

            # combine each line as a num_entity x dimension embedding tensor matrix
            if headword_e_combine is None:
                headword_e_combine = headword_e
            else:
                headword_e_combine = torch.cat((headword_e_combine, headword_e), dim=0)
            if relationword_e_combine is None:
                relationword_e_combine = relationword_e
            else:
                relationword_e_combine = torch.cat((relationword_e_combine, relationword_e), dim=0)
            if tailword_e_combine is None:
                tailword_e_combine = tailword_e
            else:
                tailword_e_combine = torch.cat((tailword_e_combine, tailword_e), dim=0)

        # now get the full embedding of head, rel and tail, i.e. head(/relation/tail)word_e_combine
        # size of each combination is : batch size x dimension
        # add each word embeddings to each corresponding entity to construct a new embedding
        # that includes both word weights and entity weights
        if head_e.size() == headword_e_combine.size():
            head_e = head_e + headword_e_combine
            tail_e = tail_e + tailword_e_combine
            rel_e = rel_e + relationword_e_combine
        #
        head_e = self.conv1(head_e, edge_index)
        head_e= f.relu(head_e)
        head_e = f.dropout(head_e, training=self.training)
        head_e = self.conv2(head_e, edge_index)
        head_e = f.relu(head_e)
        head_e = self.conv3(head_e, edge_index)

        tail_e = self.conv1(tail_e, edge_index)
        tail_e = f.relu(tail_e)
        tail_e = f.dropout(tail_e, training=self.training)
        tail_e = self.conv2(tail_e, edge_index)
        tail_e = f.relu(tail_e)
        tail_e = self.conv3(tail_e, edge_index)


        if flag == 'training':
            score = -torch.norm(head_e + rel_e - tail_e, dim=1)
            score = score + h_bias + t_bias + r_bias

        else:
            pass

        return score







