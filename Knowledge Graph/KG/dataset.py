"""
Laboratory Artificial Intelligence: Deep Learning Lab
Created on Jun 7, 2023
@Team: 02 Jiwei Pan, Ziming Fang
"""

import pickle as pkl
import numpy as np
from config import *

# Function split triples:
def read_dataset(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as file:
        lines = file.readlines()
    dataset = []
    for line in lines:
        triple = line.strip().split('\t')
        dataset.append(triple)
    return dataset

class KG(torch.utils.data.Dataset):
    def __init__(self, dataDir):
        # Base KG with entity and relation lists
        self.dataDir = dataDir

        # length of entity, relation, word, use to initialize nn.embedding
        self.lenent = []        # entity dict size
        self.lenrel = []        # relation dict size
        self.lenword = []       # word dict size

        self.entity2id = []
        self.relation2id = []
        self.word2id = []

        # First load entities and relations from the dictionaries
        file_path = os.path.join(self.dataDir, 'entities' + ".dict")

        with open(file_path, "rb") as in_file:
            self.entity2id = pkl.load(in_file)

        file_path = os.path.join(self.dataDir, 'relations' + ".dict")
        with open(file_path, "rb") as in_file:
            self.relation2id = pkl.load(in_file)

        file_path = os.path.join(self.dataDir, 'word' + ".dict")
        with open(file_path, "rb") as in_file:
            self.word2id = pkl.load(in_file)

        self.lenent = len(self.entity2id)
        self.lenrel = len(self.relation2id)
        self.lenword = len(self.word2id)

    def get_entity_relation(self):
        return self.entity2id, self.relation2id

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.lenent

# Initialize the head, relation, tail of the training
class KGTrain(KG):
    def __init__(self, negative_sample_size=2, dataDir = '', file_name = ''):
        KG.__init__(self,
                    dataDir)
        print(dataDir)
        # Initialize the data and label list
        self.negative_sample_size = negative_sample_size
        self.negmode = 0  # 0: head corruption, 1:tail corruption

        self.traintriples = []
        self.lentr = []

        file_path = os.path.join(self.dataDir, file_name)
        with open(file_path, "rb") as in_file:
            self.traintriples = pkl.load(in_file)

        self.lentr = len(self.traintriples)

    def negative_sample_generator(self, head):
        # randomly choose one different head as negative sample
        negative_samplen = np.random.randint(self.lenent, size=len(head) * self.negative_sample_size)
        negative_sample_split = np.array_split(negative_samplen, self.negative_sample_size)
        return negative_sample_split

    def negative_head_sample_generator(self, head):
        # randomly choose one different head as negative sample
        negative_samplen = np.random.randint(self.lenent, size=len(head) * self.negative_sample_size)
        return negative_samplen

    def negative_tail_sample_generator(self, tail):
        negative_samplen = np.random.randint(self.lenent, size=len(tail) * self.negative_sample_size)
        return negative_samplen

    def __getitem__(self, index):
        positive_sample = self.traintriples[index, :]
        
        headn, relationn, tailn = positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]
        negative_samplen = np.array(self.negative_sample_generator(headn))

        # From numpy to tensor ...
        head = torch.LongTensor(headn).to(device)
        relation = torch.LongTensor(relationn).to(device)
        tail = torch.LongTensor(tailn).to(device)
        negative_sample = torch.LongTensor(negative_samplen).to(device)

        return head, relation, tail, negative_sample, self.lenent, self.lenrel, self.lenword  # , htneg

    def __len__(self):
        return self.lentr

# Initialize the head, relation, tail of the testing
class KGVT(KG):
    def __init__(self, dataDir = '', file_name = ''):
        KG.__init__(self,
                    dataDir)
        # Initialize the data and label list

        self.validtriples = []
        self.lenv = []

        file_path = os.path.join(self.dataDir, file_name)
        with open(file_path, "rb") as in_file:
            self.validtriples = pkl.load(in_file)

        self.lenv = len(self.validtriples)

    def __getitem__(self, index):
        positive_sample = self.validtriples[index, :]
        headn, relationn, tailn = positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]
        # From numpy to tensor ...
        head = torch.LongTensor(headn).to(device)
        relation = torch.LongTensor(relationn).to(device)
        tail = torch.LongTensor(tailn).to(device)
        return head, relation, tail

    def __len__(self):
        return self.lenv

# Intialize head_word, relation_word, tail_word
class KGWord(KG):
    def __init__(self, dataDir='', file_name=''):
        KG.__init__(self,
                    dataDir)
        print(dataDir)
        # Initialize the data and label list

        self.trainwordtriples = []
        self.lentr = []

        file_path = os.path.join(self.dataDir, file_name)
        # file_path = datasets/ourdataset01/trainword2idx.txt.pickle
        with open(file_path, "rb") as in_file:
            self.trainwordtriples = pkl.load(in_file)

        self.lentr = len(self.trainwordtriples)

    def __getitem__(self, index):
        select_word_sample = [self.trainwordtriples[i] for i in index]
        # head, relation, tail are still lists

        headword_l = [sample[0] for sample in select_word_sample]
        relationword_l = [sample[1] for sample in select_word_sample]
        tailword_l = [sample[2] for sample in select_word_sample]

        return headword_l, relationword_l, tailword_l  # , htneg

    def getallitem(self):
        word_sample = self.trainwordtriples
        headword_l_all = [sample[0] for sample in word_sample]
        relationwaor_l_all = [sample[1] for sample in word_sample]
        tailword_l_all = [sample[2] for sample in word_sample]
        return headword_l_all, relationwaor_l_all, tailword_l_all



    def __len__(self):
        return self.lentr