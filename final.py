import torch
import torch.nn as nn
import os
import pickle as pkl
import numpy as np
from dataset import *
from model import *

class KG(torch.utils.data.Dataset):
# 接收一个参数dataDir，表示数据所在的目录
    def __init__(self, dataDir):
        # Base KG with entity and relation lists
        # 定义了一些实例变量，包括dataDir（保存数据目录路径）
        self.dataDir = dataDir
        # lenent（实体数量）
        self.lenent = []
        # lenrel（关系数量）
        self.lenrel = []

        self.entity2id = []
        self.relation2id = []


        # First load entities and relations from the dictionaries
        # 构造实体字典文件路径：通过将数据目录路径self.dataDir与文件名"entities.dict"拼接而成。
        file_path = os.path.join(self.dataDir, 'entities' + ".dict")
# 打开实体字典文件：使用open函数以二进制读取模式("rb")打开实体字典文件。
        with open(file_path, "rb") as in_file:
            self.entity2id = pkl.load(in_file)
# 从文件中加载实体数据：使用pkl.load函数从打开的文件中加载实体数据，并将结果存储在self.entity2id列表中。
        file_path = os.path.join(self.dataDir, 'relations' + ".dict")
        with open(file_path, "rb") as in_file:
            self.relation2id = pkl.load(in_file)
# 通过获取self.entity2id和self.relation2id列表的长度
        self.lenent = len(self.entity2id)
        self.lenrel = len(self.relation2id)
# 类中存储实体和关系的字典或列表
    def get_entity_relation(self):
        return self.entity2id, self.relation2id
# __getitem__方法定义了通过索引访问类实例时的行为
    def __getitem__(self, index):
        pass
# __len__方法定义了获取类实例长度的行为
    def __len__(self):
        return self.lenent

# 继承自KG类
class KGTrain(KG):
    # negative_sample_size用于指定负样本的数量
    def __init__(self, negative_sample_size=2, dataDir = '', file_name = ''):
        KG.__init__(self,dataDir)
        print(dataDir)
        # Initialize the data and label list
        self.negative_sample_size = negative_sample_size
        # negmode表示负样本的生成方式（0表示head替换，1表示tail替换)
        self.negmode = 0  # 0: head corruption, 1:tail corruption

        self.traintriples = []
        self.lentr = []
# 通过读取指定路径下的文件，加载了训练三元组数据到self.traintriples列表中。
        file_path = os.path.join(self.dataDir, file_name)
        with open(file_path, "rb") as in_file:
            self.traintriples = pkl.load(in_file)
# self.lentr则记录了训练三元组的数量。
        self.lentr = len(self.traintriples)
# 它接受参数head作为输入。该方法用于生成负样本。
    def negative_sample_generator(self, head):
     # 首先使用np.random.randint函数生成一个随机整数数组negative_samplen
     # 数组的长度为len(head) * self.negative_sample_size
     # self.lenent表示实体的总数，self.negative_sample_size表示负样本的数量。
        negative_samplen = np.random.randint(self.lenent, size=len(head) * self.negative_sample_size)
        return negative_samplen

    def __getitem__(self, index):
        positive_sample = self.traintriples[index, :]
# 首先从self.traintriples中获取索引为index的正样本，并将其拆分为headn、relationn和tailn三个数组。

        headn, relationn, tailn = positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]
        # 生成负样本的索引数组negative_samplen。
        negative_samplen = self.negative_sample_generator(headn)
# 转换为torch.LongTensor类型，以便进行后续的深度学习模型训练。
        # From numpy to tensor ...
        head = torch.LongTensor(headn)
        relation = torch.LongTensor(relationn)
        tail = torch.LongTensor(tailn)
        negative_sample = torch.LongTensor(negative_samplen)
# self.lenent表示实体的总数。
        return head, relation, tail, negative_sample, self.lenent  # , htneg

    def __len__(self):
        return self.lentr

# 表示知识图谱的验证集
class KGVT(KG):
    def __init__(self, dataDir = '', file_name = ''):
       # 调用父类KG的__init__方法来初始化数据目录。
        KG.__init__(self,dataDir)
        # Initialize the data and label list

        self.validtriples = []
        self.lenv = []
# 根据给定的数据目录和文件名，读取验证集数据并存储在self.validtriples中
        file_path = os.path.join(self.dataDir, file_name)
        with open(file_path, "rb") as in_file:
            self.validtriples = pkl.load(in_file)

        self.lenv = len(self.validtriples)

    def __getitem__(self, index):
        # 首先从self.validtriples中获取索引为index的正样本
        positive_sample = self.validtriples[index, :]
        # 并将其拆分为headn、relationn和tailn三个数组。
        headn, relationn, tailn = positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]
        # TODO add corruption here?
        # From numpy to tensor ...
        head = torch.LongTensor(headn)
        relation = torch.LongTensor(relationn)
        tail = torch.LongTensor(tailn)
        return head, relation, tail

    def __len__(self):
        return self.lenv
# Whole Class with additions:
class KGE(nn.Module):
    def __init__(self, model_name, nentity, nrelation, dimension):
        super(KGE, self).__init__()
        # Define Hyperparameters
        self.dimension = dimension
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation

        # embeddings: Assign one vector to each of the entities and relations, the vectors will be randomly initialized and then trained
        self.entity_embedding = nn.Embedding(self.nentity, self.dimension)
        # adding two embeddings for each relation, one for original and the other for the reverse counterpart
        self.entity_embedding_max_norm = nn.Embedding(self.nentity, self.dimension, max_norm=1)
        self.relation_embedding = nn.Embedding(2*self.nrelation, self.dimension)
        self.init_size = 0.001

        self.initialization()

    def initialization(self):
        self.entity_embedding.weight.data = self.init_size * torch.randn((self.nentity, self.dimension))
        self.entity_embedding_max_norm.weight.data = self.init_size * torch.randn((self.nentity, self.dimension))
        self.relation_embedding.weight.data = self.init_size * torch.randn((2*self.nrelation, self.dimension))

    def forward(self, head, relation, tail):
        pass

    def marginrankingloss(self, positive_score, negative_score, margin):
        rl = nn.ReLU()
        l = torch.sum(rl(negative_score - positive_score + torch.tensor([margin])))
        return l


class TransE(KGE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, head, relation, tail):
        KGE.__init__(self, TransE, len(head), len(relation), dimension=60)
        self.head = head
        self.tail = tail
        self.relation = relation

    def score(self, head, relation, tail, flag):
        #TransE aims to hold h + r - t = 0, so the distance is ||h + r - t||
        if flag == 'training':
          head_e = self.entity_embedding(head)
          rel_e = self.relation_embedding(relation)
          tail_e = self.entity_embedding(tail)
          score = -torch.norm(head_e + rel_e - tail_e, dim=1)
        else:
          pass

        return score
#%%

# def reverse_adding(head_f, tail_f, relation_f, ent_num_f):
#     reverse_relation_f = relation_f + ent_num_f
#     reversed_head_f = tail_f
#     reversed_tail_f = head_f
#     head_augment = torch.cat((head, reversed_head_f))
#     relation_augment = torch.cat((relation, reverse_relation_f))
#     tail_augment = torch.cat((tail, reversed_tail_f))
#     return head_augment, relation_augment, tail_augment

if __name__ == '__main__':
    dataset_name = "dataset"
    # data_path，根据数据集名称拼接而成。
    data_path = 'C://Users//panji//OneDrive//Desktop//lab//4-DataPreprocessing//' + dataset_name
    # 这段代码假设数据集文件是使用Python的pickle模块进行序列化后保存的。
    # 这段代码假设数据集文件是使用Python的pickle模块进行序列化后保存的。
    # Pickle是Python中用于序列化和反序列化对象的标准模块。它可以将对象转换为字节流进行保存，然后在需要时重新加载回对象。
    train_file_name = 'train.txt' + ".pickle"
    valid_file_name = 'valid.txt' + ".pickle"
    test_file_name = 'test.txt' + ".pickle"
    train = KGTrain(2, data_path, train_file_name)
    valid = KGVT(data_path, valid_file_name)
    test = KGVT(data_path, test_file_name)
    head, relation, tail, _, ent_num = train[:]

    reverse_relation = relation + ent_num
    reversed_head = tail
    reversed_tail = head
    head = torch.cat((head, reversed_head))
    relation = torch.cat((relation, reverse_relation))
    tail = torch.cat((tail, reversed_tail))

    # augmentation train

    print(os.getcwd())

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100
    head_v, relation_v, tail_v = valid[:]
    head_t, relation_t, tail_t = test[:]

    transe = TransE(head, relation, tail)
    nrelation = max(relation) + 1
    nentity = max(max(tail), max(head)) + 1

    # optimizer
    learning_rate = .0001
    optimizer = torch.optim.Adam(transe.parameters(), lr=learning_rate)

    # training
    idx = torch.arange(0, len(train), 1)
    if True:
        train_loader = torch.utils.data.DataLoader(
            idx,
            batch_size=128,
            shuffle=True,
        )

        print('training')
        margin = 1
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            totalloss = 0
            for i, bidx in enumerate(train_loader):
                head, relation, tail, negsamp, _ = train[bidx]
                relationInv = relation + nrelation

                # Forward pass
                head_neg = negsamp[:negsamp.size()[0] // 2]
                tail_neg = negsamp[negsamp.size()[0] // 2:]

                positive_score = transe.score(head, relation, tail, 'training')
                negative_score_tail = transe.score(head, relation, tail_neg, 'training')
                negative_score_head = transe.score(head_neg, relation, tail, 'training')

                loss = transe.marginrankingloss(positive_score, negative_score_head, margin) / len(negative_score_head)
                loss = loss + transe.marginrankingloss(positive_score, negative_score_tail, margin) / len(
                    negative_score_tail)

                # Adding inverse of bach to the training
                positive_score_inv = transe.score(tail, relationInv, head, 'training')
                negative_score_tail_inv = transe.score(tail_neg, relationInv, head, 'training')
                negative_score_head_inv = transe.score(tail, relationInv, head_neg, 'training')

                loss += transe.marginrankingloss(positive_score_inv, negative_score_head_inv, margin) / len(
                    negative_score_head_inv)
                loss += transe.marginrankingloss(positive_score_inv, negative_score_tail_inv, margin) / len(
                    negative_score_tail_inv)

                totalloss += loss

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if (i + 1) % 2 == 0:
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            print(totalloss)

        # Save the model checkpoint
        torch.save(transe.state_dict(), 'transe-skg.ckpt')
        # to load : model.load_state_dict(torch.load(save_name_ori))

    transe.load_state_dict(torch.load('transe-skg.ckpt'))

    if True:
        # Test the model
        head, relation, tail = test[:]
        ranks = torch.ones(len(head))
        all_entity = torch.LongTensor(np.arange(0, nentity))
        head_expand = torch.ones(nentity)
        tail_expand = torch.ones(nentity)
        relation_expand = torch.ones(nentity)

        total_rank = 0
        total_rank_Inv = 0
        total_rank_OI = 0

        mean_rank = {}
        hits10 = 0
        hits10_Inv = 0
        hits10_OI = 0

        with torch.no_grad():
            for idx in range(len(head)):
                h, r, t = head[idx] * head_expand, relation[idx] * relation_expand, tail[idx] * tail_expand
                rInv = r + nrelation
                h, r, t, rInv = h.type(torch.LongTensor), r.type(torch.LongTensor), t.type(torch.LongTensor), rInv.type(
                    torch.LongTensor)

                Corrupted_score_tail = transe.score(h, r, all_entity, 'training')
                Corrupted_score_head = transe.score(all_entity, r, t, 'training')

                argsort_tail = torch.argsort(Corrupted_score_tail, dim=0, descending=True)
                argsort_head = torch.argsort(Corrupted_score_head, dim=0, descending=True)

                ranking_tail = (argsort_tail == t).nonzero(as_tuple=True)[0]
                ranking_head = (argsort_head == h).nonzero(as_tuple=True)[0]
                print(ranking_tail)

                avg_rank = (ranking_head + ranking_tail) / 2
                total_rank = total_rank + avg_rank
                hits10 += 1 if avg_rank < 11 else 0

                # Prediction based on reverse f(h,r,t) -> f(t,r-1,h)
                Corrupted_score_tail_Inv = transe.score(all_entity, rInv, h, 'training')
                Corrupted_score_head_Inv = transe.score(t, rInv, all_entity, 'training')

                argsort_tail_Inv = torch.argsort(Corrupted_score_tail_Inv, dim=0, descending=True)
                argsort_head_Inv = torch.argsort(Corrupted_score_head_Inv, dim=0, descending=True)

                ranking_tail_Inv = (argsort_tail_Inv == t).nonzero(as_tuple=True)[0]
                ranking_head_Inv = (argsort_head_Inv == h).nonzero(as_tuple=True)[0]
                print(ranking_tail_Inv)

                avg_rank_Inv = (ranking_head_Inv + ranking_tail_Inv) / 2
                total_rank_Inv = total_rank_Inv + avg_rank_Inv
                hits10_Inv += 1 if avg_rank_Inv < 11 else 0

                # Prediction based on average of reverse and original f(h,r,t) -> (f(t,r-1,h) + f(h,r,t))/2
                Corrupted_score_tail_OI = (transe.score(all_entity, rInv, h, 'training') + transe.score(h, r,
                                                                                                        all_entity,
                                                                                                        'training')) / 2
                Corrupted_score_head_OI = (transe.score(t, rInv, all_entity, 'training') + transe.score(all_entity, r,
                                                                                                        t,
                                                                                                        'training')) / 2

                argsort_tail_OI = torch.argsort(Corrupted_score_tail_OI, dim=0, descending=True)
                argsort_head_OI = torch.argsort(Corrupted_score_head_OI, dim=0, descending=True)

                ranking_tail_OI = (argsort_tail_OI == t).nonzero(as_tuple=True)[0]
                ranking_head_OI = (argsort_head_OI == h).nonzero(as_tuple=True)[0]
                print(ranking_tail_OI)

                avg_rank_OI = (ranking_head_OI + ranking_tail_OI) / 2
                total_rank_OI = total_rank_OI + avg_rank_OI
                hits10_OI += 1 if avg_rank_OI < 11 else 0

                print(idx, len(head), hits10)

        mean_rank = total_rank / len(head)
        mean_rank_Inv = total_rank_Inv / len(head)
        mean_rank_OI = total_rank_OI / len(head)

        hits10 = hits10 / len(head)
        hits10_OI = hits10_OI / len(head)
        hits10_Inv = hits10_Inv / len(head)

        print("mean rank orig:", mean_rank, "\nhits@10 orig:", hits10)
        print("mean rank inv:", mean_rank_Inv, "\nhits@10 inv:", hits10_Inv)
        print("mean rank avg:", mean_rank_OI, "\nhits@10 avg:", hits10_OI)
    # augmentationtrain

    print(f"relation的长度：{len(relation)}")

    #original possitive triples (h,r,t)
    # head = torch.LongTensor([0, 1, 2, 3, 4])
    # tail = torch.LongTensor([4, 3, 2, 1, 0])
    # relation = torch.LongTensor([1, 2, 0, 1, 2])
    # reverserelation = torch.LongTensor([1, 2, 0, 1, 2]) + torch.LongTensor([3])
    print(" relation: ", relation, "\n reverse relation:", reverse_relation)
    print(" new head: ", head)


    print("\n Positive Triples: \n ", torch.stack((head, relation, tail)), "\n")

    # Corruptedtail for generating negative sample (h,r,t')
    tailc = torch.randint(0, 5, tail.size())

    print("Negative Triples: \n ", torch.stack((head, relation, tailc)), "\n")

    transe = TransE(head, relation, tail)
    p_score = transe.score(head, relation, tail, 'training')
    n_score = transe.score(head, relation, tailc, 'training')

    # (Positive triples score, Negative triples score)
    print("Positive triples score =\n" , p_score, "\n\n" ,"Negative triples score = \n", n_score, "\n")

    print("Total loss is: ", transe.marginrankingloss(p_score, n_score, margin = 2))
    transe = TransE(head, relation, tail)

    for i in range(transe.nentity):
        entity = transe.entity_embedding(torch.tensor([i]))
        entity_max_norm = transe.entity_embedding_max_norm(torch.tensor([i]))
        entity_normalized = torch.nn.functional.normalize(entity)
        print("\nEntity" + str(i) + ":", torch.norm(entity), "\t Entity_Normalized" + str(i) + ":",
              torch.norm(entity_normalized), "\t Entity_Max_Norm" + str(i) + ":", torch.norm(entity_max_norm), )