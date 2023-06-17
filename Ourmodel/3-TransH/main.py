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
from model import *
from metrics import *
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import StepLR
import time


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
    valid_file_name = 'valid.txt' + ".pickle"
    test_file_name = 'test.txt' + ".pickle"
    train_word_file_name = 'trainword2idx.txt' + ".pickle"
    test_word_file_name = 'testword2idx.txt' + ".pickle"
    valid_word_file_name = 'validword2idx.txt' + ".pickle"

    training = KGTrain(2, data_path, train_file_name)
    valid = KGVT(data_path, valid_file_name)
    testing = KGVT(data_path, test_file_name)
    training_word = KGWord(data_path, train_word_file_name)
    valid_word = KGWord(data_path, valid_word_file_name)
    testing_word = KGWord(data_path, test_word_file_name)

    # ent_num, rel_num, word_num are total numbers of entities and words in dictionary, not batch or .txt.pickle line number
    head, relation, tail, neg, ent_num, rel_num, word_num = training[:]
    # v: valid t: test w: word, all are tensor value
    head_v, relation_v, tail_v = valid[:]
    head_t, relation_t, tail_t = testing[:]
    # to be tested, remind myself later, the index maybe incorrect
    head_w, relation_w, tail_w = training_word.getallitem()
    head_t_w, relation_t_w, tail_t_w = testing_word.getallitem()
    head_v_w, relation_v_w, tail_v_w = valid_word.getallitem()

    kge_model = TransH(head, relation, tail, ent_num, rel_num, word_num).to(device)

    # optimizer
    learning_rate = .005
    optimizer = torch.optim.Adam(kge_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=250, gamma=1)

    # training
    idx = torch.arange(0, len(training), 1)


    if True:
        num_workers = 4
        train_loader = torch.utils.data.DataLoader(
            idx,
            batch_size=128,
            shuffle=True,
            num_workers=num_workers
        )

        print('training')
        margin = torch.tensor([1]).to(device)
        total_step = len(train_loader)

        # Early stopping variables
        best_val_loss = float('inf')
        patience = 3
        counter = 0

        for epoch in range(num_epochs):
            totalloss = 0
            start_time = time.time()
            for i, bidx in enumerate(train_loader):
                # head, relation, tail are tensor
                head, relation, tail, negsamp, lenent, lenrel, lenword = training[bidx]
                # print(bidx)
                # print(bidx.type)
                head_w, relation_w, tail_w = training_word[bidx]
                # word embedding

                # Forward pass
                head_neg = negsamp[:negsamp.size()[0] // 2]
                tail_neg = negsamp[negsamp.size()[0] // 2:]
                # head: (bidx x 1) head_w: [(bidx lists)]
                positive_score = kge_model.score(head, relation, tail, head_w, relation_w, tail_w, 'training').to(device)
                negative_score_tail = kge_model.score(head, relation, tail_neg, head_w, relation_w, tail_w, 'training').to(device)
                negative_score_head = kge_model.score(head_neg, relation, tail, head_w, relation_w, tail_w, 'training').to(device)

                loss = kge_model.marginrankingloss(positive_score, negative_score_head, margin) / len(
                    negative_score_head)
                loss = loss + kge_model.marginrankingloss(positive_score, negative_score_tail, margin) / len(
                    negative_score_tail)

                totalloss += loss

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                if totalloss < best_val_loss:
                    best_val_loss = totalloss
                    counter = 0
                    bestepoch=epoch
                else:
                    counter += 1

                if counter >= patience:
                    print("Early stopping triggered. Training stopped.")
                    break

                writer.add_scalar('num_workers/num_workers', num_workers, 0)
                writer.add_scalar('learning_rate/epoch', learning_rate, epoch)
            print('Epoch:', epoch, 'Learning Rate:', scheduler.get_last_lr())


                # if (i + 1) % 2 == 0:
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                #           .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

            if epoch % 10 == 0:
                print(totalloss)
                head, relation, tail = testing[:]
                mean_rank, hits10, mrr = ranking(head, relation, tail, head_t_w, relation_t_w, tail_t_w, ent_num, kge_model)

                print('Epoch:', epoch, 'Hits@10:', hits10, 'MRR:',mrr)


            writer.add_scalar('Loss/train', totalloss, epoch)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time with num_workers={num_workers}: {elapsed_time} seconds")

            print()

            if counter >= patience:
                break




        # Save the model checkpoint
        torch.save(kge_model.state_dict(), 'transe-skg.ckpt')
        # to load : model.load_state_dict(torch.load(save_name_ori))

    kge_model.load_state_dict(torch.load('transe-skg.ckpt'))

    if True:
        # Test the model
        head, relation, tail = testing[:]
        mean_rank, hits10, mrr = ranking(head, relation, tail, head_t_w, relation_t_w, tail_t_w, ent_num, kge_model)

        writer.add_scalar('Rank/MeanRank', mean_rank, 0)
        writer.add_scalar('mrr/MRR', mrr, 0)
        writer.add_scalar('Hits/Hits@10', hits10, 0)

        # print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
    writer.close()
    print(f"best epoch{bestepoch}")