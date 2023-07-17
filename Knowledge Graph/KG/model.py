"""
Laboratory Artificial Intelligence: Deep Learning Lab
Created on Jun 7, 2023
@Team: 02 Jiwei Pan, Ziming Fang
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
from config import device
import config


class KGE(nn.Module):
    def __init__(self, model_name, nentity, nrelation, nword, dimension):
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

    # lossfunction
    def marginrankingloss(self, positive_score, negative_score, margin):
        rl = nn.ReLU().cuda()
        #print("11111")
        l = torch.sum(rl(negative_score - positive_score + margin))
        return l

    def hardmarginloss(self, positive_score, negative_score):
        rl = nn.ReLU().cuda()
        positive_sample_loss = torch.sum(rl(- positive_score - self.margin))
        negative_sample_loss = torch.sum(rl(negative_score + self.margin_neg))
        l = (positive_sample_loss + negative_sample_loss) / 2
        return l

    def loglikelihoodloss(self, positive_score, negative_score):
        positive_sample_loss = -f.logsigmoid(self.margin + positive_score).mean()
        negative_sample_loss = -f.logsigmoid(-negative_score - self.margin).mean()
        l = (positive_sample_loss + negative_sample_loss) / 2
        return l


# TransE model
class TransE(KGE):

    def __init__(self, head, relation, tail, num_ent, num_rel, num_word):
        KGE.__init__(self, TransH, num_ent, num_rel, num_word, dimension=60)
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

    def score(self, head, relation, tail, head_w, rel_w, tail_w, flag):
        # TransE aims to hold h + r - t = 0, so the distance is ||h + r - t||
        if flag == 'training':
            head_e = self.entity_embedding(head)
            rel_e = self.relation_embedding(relation)
            tail_e = self.entity_embedding(tail)

            head_e = torch.nn.functional.normalize(head_e)
            rel_e = torch.nn.functional.normalize(rel_e)
            tail_e = torch.nn.functional.normalize(tail_e)
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
                headword_t = torch.tensor(head_w[i]).to(device)
                relationword_t = torch.tensor(rel_w[i]).to(device)
                tailword_t = torch.tensor(tail_w[i]).to(device)

                # embed words line by line since different dim (different word length of each entity)
                headword_e = self.word_embedding(headword_t)
                relationword_e = self.word_embedding(relationword_t)
                tailword_e = self.word_embedding(tailword_t)

                headword_e = torch.nn.functional.normalize(headword_e)
                relationword_e = torch.nn.functional.normalize(relationword_e)
                tailword_e = torch.nn.functional.normalize(tailword_e)

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
            score = -torch.norm(head_e + rel_e - tail_e, dim=1, p=2)
            score = score + h_bias + t_bias + r_bias

        else:
            pass

        return score


# TransH model
class TransH(KGE):

    def __init__(self, head, relation, tail, num_ent, num_rel, num_word):
        KGE.__init__(self, TransH, num_ent, num_rel, num_word, dimension=60)
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

    def score(self, head, relation, tail, head_w, rel_w, tail_w, flag):
        #TransE aims to hold h + r - t = 0, so the distance is ||h + r - t||
        if flag == 'training':          
            head_e = self.entity_embedding(head)
            rel_e = self.relation_embedding(relation)
            tail_e = self.entity_embedding(tail)

            head_e = torch.nn.functional.normalize(head_e)
            rel_e = torch.nn.functional.normalize(rel_e)
            tail_e = torch.nn.functional.normalize(tail_e)
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
                headword_t = torch.tensor(head_w[i]).to(device)
                relationword_t = torch.tensor(rel_w[i]).to(device)
                tailword_t = torch.tensor(tail_w[i]).to(device)

                # embed words line by line since different dim (different word length of each entity)
                headword_e = self.word_embedding(headword_t)
                relationword_e = self.word_embedding(relationword_t)
                tailword_e = self.word_embedding(tailword_t)

                headword_e = torch.nn.functional.normalize(headword_e)
                relationword_e= torch.nn.functional.normalize( relationword_e)
                tailword_e = torch.nn.functional.normalize( tailword_e)


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

            # compute projection of relation
            rel_e_proj = self.relation_projection(relation)
            rel_e_proj = f.normalize(rel_e_proj, p=2, dim=1)

            head_e_proj = head_e - (rel_e_proj * head_e).sum(dim = 1).unsqueeze(1) * rel_e_proj
            tail_e_proj = tail_e - (rel_e_proj * tail_e).sum(dim = 1).unsqueeze(1) * rel_e_proj

            score = -torch.norm(head_e_proj + rel_e - tail_e_proj, dim=1)

            score = score + h_bias + t_bias + r_bias

        else: 
          pass

        return score






