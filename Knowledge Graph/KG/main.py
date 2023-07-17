"""
Laboratory Artificial Intelligence: Deep Learning Lab
Created on Jun 7, 2023
@Team: 02
"""

from torch.utils.tensorboard import SummaryWriter
from dataset import *
from model import *
from metrics import *
from torch.utils.data import DataLoader
import os
from torch.optim.lr_scheduler import StepLR
import time
import random


if __name__ == '__main__':
    # use Tensorboard
    writer = SummaryWriter(log_dir='logs')
    print(os.getcwd())
    print(device)

    # epoch
    num_epochs = 100

    # Load dataset
    dataset_name = "new dataset"
    data_path = 'C:/Users/panji/OneDrive/Desktop/Knowledge Graph/KG/datasets/' + dataset_name
    train_file_name = 'train.txt' + ".pickle"
    valid_file_name = 'valid.txt' + ".pickle"
    test_file_name = 'test.txt' + ".pickle"
    train_word_file_name = 'trainword2idx.txt' + ".pickle"
    test_word_file_name = 'testword2idx.txt' + ".pickle"
    valid_word_file_name = 'validword2idx.txt' + ".pickle"
    neg_sample_size = 1

    # Initialize training set, test set, validation set
    training = KGTrain(neg_sample_size, data_path, train_file_name)
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

    # initial TransE model
    kge_model = TransE(head, relation, tail, ent_num, rel_num, word_num).to(device)

    # optimizer
    learning_rate = .001
    optimizer = torch.optim.Adam(kge_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=250, gamma=1)

    # training
    idx = torch.arange(0, len(training), 1)
    if True:
        num_workers = 2
        train_loader = torch.utils.data.DataLoader(
            idx,
            batch_size=128,
            shuffle=True,
            num_workers=num_workers
        )

        print('training')

        print(f'negative sample size: {neg_sample_size}')
        # γ： margin
        margin = torch.tensor([1]).to(device)
        total_step = len(train_loader)
        # Early stopping variables
        best_val_loss = float('inf')
        patience = 3
        counter = 0

        # Forward pass
        for epoch in range(num_epochs):
            totalloss = 0
            start_time = time.time()
            for i, bidx in enumerate(train_loader):
                # head, relation, tail
                head, relation, tail, negsamp, lenent, lenrel, lenword = training[bidx]
                # head_word, relation_word, tail_word
                head_w, relation_w, tail_w = training_word[bidx]

                positive_score = kge_model.score(head, relation, tail, head_w, relation_w, tail_w, 'training').to(device)
                loss = 0

                # Generate negative samples of head and tail
                for neg_idx in range(neg_sample_size):
                    tail_neg_each = negsamp[neg_idx]
                    head_neg_each = negsamp[neg_idx]
                    tail_w_neg = tail_w
                    head_w_neg = head_w
                    random.shuffle(tail_w_neg)
                    random.shuffle(head_w_neg)
                    negative_score_tail_each = kge_model.score(head, relation, tail_neg_each, head_w, relation_w, tail_w_neg, 'training').to(device)
                    negative_score_head_each = kge_model.score(head_neg_each, relation, tail, head_w_neg, relation_w, tail_w, 'training').to(device)

                # loss function
                    # marginrankingloss
                    loss = kge_model.marginrankingloss(positive_score, negative_score_tail_each, margin)/len(negative_score_tail_each)
                    loss = loss + kge_model.marginrankingloss(positive_score, negative_score_head_each, margin)/len(negative_score_head_each)

                    totalloss += loss

                    # hardmarginloss
                    # loss = kge_model.hardmarginloss(positive_score, negative_score_tail_each) / len(
                    #     negative_score_tail_each)
                    # loss = loss + kge_model.hardmarginloss(positive_score, negative_score_head_each) / len(
                    #     negative_score_head_each)
                    #
                    # totalloss += loss

                    # loglikelihoodloss
                    # loss = kge_model.loglikelihoodloss(positive_score, negative_score_tail_each) / len(
                    #     negative_score_tail_each)
                    # loss = loss + kge_model.loglikelihoodloss(positive_score, negative_score_head_each) / len(
                    #     negative_score_head_each)
                    #
                    # totalloss += loss

            # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Early stopping
            with torch.no_grad():
                if totalloss < best_val_loss:
                    best_val_loss = totalloss
                    counter = 0
                    bestepoch = epoch
                else:
                    counter += 1
                if counter >= patience:
                    print("Early stopping triggered. Training stopped.")
                    break

                writer.add_scalar('num_workers/num_workers', num_workers, 0)
                writer.add_scalar('learning_rate/epoch', learning_rate, epoch)
            print('Epoch:', epoch, 'Learning Rate:', scheduler.get_last_lr())
            print(totalloss)


            #  output every 10 epoch
            if epoch % 10 == 0:
                print(totalloss)
            writer.add_scalar('Loss/train', totalloss, epoch)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time with num_workers={num_workers}: {elapsed_time} seconds")

            print()

            if counter >= patience:
                break

        # Save the model checkpoint
        torch.save(kge_model.state_dict(), 'transe.ckpt')
        # to load : model.load_state_dict(torch.load(save_name_ori))

    kge_model.load_state_dict(torch.load('transe.ckpt'))

    if True:
        # Test the model
        head, relation, tail = testing[:]
        mean_rank, hits10, mrr = ranking(head, relation, tail, head_t_w, relation_t_w, tail_t_w, ent_num, kge_model)
        writer.add_scalar('Rank/MeanRank', mean_rank, 0)
        writer.add_scalar('Hits/Hits@10', hits10, 0)
        writer.add_scalar('mrr/MRR', mrr, 0)

    writer.close()
    print(f"best epoch{bestepoch}")
