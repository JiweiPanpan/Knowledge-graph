"""
Laboratory Artificial Intelligence: Deep Learning Lab
Created on Jun 7, 2023
@Team: 02
"""

from ast import arg


import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataset import *
from metrics import *
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import StepLR
import time
import networkx as nx
import matplotlib.pyplot as plt

from model_GCN import *
import torch
from transformers import BertModel, BertTokenizer
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as f
# 创建一个空的图


from config import device

if __name__ == '__main__':

    writer = SummaryWriter(log_dir='logs')
    print(os.getcwd())

    # Device configuration
    device = torch.device('cuda:0')
    num_epochs = 100

    # Load dataset
    dataset_name = "new dataset"
    data_path = 'datasets/' + dataset_name
    train_file_name = 'train.txt' + ".pickle"
    test_file_name = 'test.txt' + ".pickle"
    train_word_file_name = 'trainword2idx.txt' + ".pickle"
    test_word_file_name = 'testword2idx.txt' + ".pickle"
    # for KG
    dataset_train = 'C:/Users/panji/OneDrive/Desktop/Ourmodel/3-TransH/datasets/new dataset/train.txt'  # 数据集文件路径
    dataset_test = 'C:/Users/panji/OneDrive/Desktop/Ourmodel/3-TransH/datasets/new dataset/test.txt'  # 数据集文件路径

    # entities and word
    training = KGTrain(2, data_path, train_file_name)
    testing = KGVT(data_path, test_file_name)
    training_word = KGWord(data_path, train_word_file_name)
    testing_word = KGWord(data_path, test_word_file_name)

    head, relation, tail, neg, ent_num, rel_num, word_num = training[:]
    head_t, relation_t, tail_t = testing[:]
    head_w, relation_w, tail_w = training_word.getallitem()
    head_t_w, relation_t_w, tail_t_w = testing_word.getallitem()
    # for KG
    dataset_train = training.read_dataset(dataset_train)
    dataset_test = testing.read_dataset(dataset_test)
    # ent_num, rel_num, word_num are total numbers of entities and words in dictionary, not batch or .txt.pickle line number

    # split entities and word
    # num_samples = len(dataset)
    # split_index = int(0.8 * num_samples)
    # train_head_e = head[:split_index]
    # test_head_e = head[split_index:]
    # train_tail_e = tail[:split_index]
    # test_tail_e = tail[split_index:]
    # train_rel_e = relation[:split_index]
    # test_rel_e = relation[split_index:]
    #
    # num_samples = len(dataset)
    # split_index = int(0.8 * num_samples)
    # train_head_w = head_w[:split_index]
    # test_head_w = head_w[split_index:]
    # train_tail_w = tail_w[:split_index]
    # test_tail_w = tail_w[split_index:]
    # train_rel_w = relation_w[:split_index]
    # test_rel_w = relation_w[split_index:]
    input_dim = 60
    hidden_dim = 64
    output_dim = 60
    dimension = 60
    kge_model = GCNTransE(head, relation, tail, ent_num, rel_num, word_num, input_dim, hidden_dim, output_dim, dimension)
    edge_index_train, num_nodes_train = edge_index(dataset_train)
    edge_index_test, num_nodes_test = edge_index(dataset_test)

    # adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    # for edge in edge_index.t():
    #     src, tgt = edge.tolist()
    #
    #     adjacency_matrix[src, tgt] = 1.0

    # data = Data(Head=Head, Tail=Tail,edge_index=edge_index, adj=adjacency_matrix, train_mask=train_mask,  test_mask=test_mask)
    # data_train = Data( edge_index=edge_index_train)

    # optimizer
    learning_rate = .005
    optimizer = torch.optim.Adam(kge_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=250, gamma=1)


    # training
    margin = torch.tensor([1])
    # Early stopping variables
    best_val_loss = float('inf')
    patience = 3
    counter = 0




    for epoch in range(num_epochs):
        # totalloss = 0
        start_time = time.time()

        # edge_index = []
        # head_features = []
        # tail_features = []
        # relation_features = []
        #
        # for triple in dataset:
        #     head, relation, tail = triple
        #     head_idx = head2id[head]
        #     tail_idx = tail2id[tail]
        #     relation_idx = relation2id[relation]

        # 添加节点特征


            # x.append(torch.cat([head, tail], dim=0))
            # head_features.append(head_w)
            # tail_features.append(tail_w)
            # relation_features.append(relation_w)
            # 添加边索引
            # edge_index.append([head_idx, tail_idx])
            # edge_index.append([tail_idx, head_idx])
            # 添加节点标签
        # split_index = int(0.8 * num_samples)
        # test_head_w = head[split_index:]
        # test_tail_w = tail[split_index:]
        # test_relation_w = relation[split_index:]

        # 转换为PyG的Data对象
        # Head = torch.stack(head_features, dim=1).squeeze(0)
        # Tail = torch.stack(tail_features, dim=1).squeeze(0)
        # Relation = torch.stack(relation_features, dim=1).squeeze(0)

        # edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 构建邻接矩阵
        # num_nodes = Head.size(0)
        # adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        # for edge in edge_index.t():
        #     src, tgt = edge.tolist()
        #
        #     adjacency_matrix[src, tgt] = 1.0
        #
        # # data = Data(Head=Head, Tail=Tail,edge_index=edge_index, adj=adjacency_matrix, train_mask=train_mask,  test_mask=test_mask)
        # data = Data(Head=Head, Tail=Tail, edge_index=edge_index)



        # Forward pass
        head_neg = neg[:neg.size()[0] // 2]
        tail_neg = neg[neg.size()[0] // 2:]
        # head: (bidx x 1) head_w: [(bidx lists)]
        positive_score = kge_model.score(head, relation, tail, head_w, relation_w, tail_w, edge_index_train, 'training')
        negative_score_tail = kge_model.score(head, relation, tail_neg, head_w, relation_w, tail_w, edge_index_train,'training')
        negative_score_head = kge_model.score(head_neg, relation, tail, head_w, relation_w, tail_w, edge_index_train,'training')

        # marginrankingloss
        # loss = kge_model.marginrankingloss(positive_score, negative_score_head, margin) / len(
        #     negative_score_head)
        loss = kge_model.marginrankingloss(positive_score, negative_score_tail, margin) / len(
            negative_score_tail)

        # hardmarginloss
        # loss = kge_model.hardmarginloss(positive_score, negative_score_head) / len(
        #     negative_score_head)
        # loss = loss + kge_model.hardmarginloss(positive_score, negative_score_tail) / len(
        #     negative_score_tail)

        # loglikelihoodloss
        # loss = kge_model.loglikelihoodloss(positive_score, negative_score_head) / len(
        #     negative_score_head)
        # loss = loss + kge_model.loglikelihoodloss(positive_score, negative_score_tail) / len(
        #     negative_score_tail)
        # totalloss += loss


        # Backward and optimize
        optimizer.zero_grad()

        # negative log-likelihood loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            if loss < best_val_loss:
                best_val_loss = loss
                counter = 0
                bestepoch=epoch
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

            # writer.add_scalar('num_workers/num_workers', num_workers, 0)
            writer.add_scalar('learning_rate/epoch', learning_rate, epoch)
        print('Epoch:', epoch, 'Learning Rate:', scheduler.get_last_lr())


                # if (i + 1) % 2 == 0:
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        if epoch % 10 == 0:
            print(loss)
            # head, relation, tail = testing[:]
            # mean_rank, hits10, mrr = ranking(head, relation, tail, head_t_w, relation_t_w, tail_t_w, ent_num, kge_model)
            #
            # print('Epoch:', epoch, 'Hits@10:', hits10, 'MRR:',mrr)


        writer.add_scalar('Loss/train', loss, epoch)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time with num_workers=: {elapsed_time} seconds")

        print()

        if counter >= patience:
            break




        # Save the model checkpoint
        torch.save(kge_model.state_dict(), 'transe-skg.ckpt')
        # to load : model.load_state_dict(torch.load(save_name_ori))

    kge_model.load_state_dict(torch.load('transe-skg.ckpt'))

    if True:
        # Test the model

        mean_rank, hits10, mrr = ranking(head_t, relation_t, tail_t, head_t_w, relation_t_w, tail_t_w, ent_num, edge_index_test , kge_model)

        writer.add_scalar('Rank/MeanRank', mean_rank, 0)
        writer.add_scalar('mrr/MRR', mrr, 0)
        writer.add_scalar('Hits/Hits@10', hits10, 0)

        # print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    writer.close()
    print(f"best epoch{bestepoch}")