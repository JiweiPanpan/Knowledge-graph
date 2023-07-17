"""
Laboratory Artificial Intelligence: Deep Learning Lab
Created on Jun 7, 2023
@Team: 02 Jiwei Pan, Ziming Fang
"""

from config import *
import torch
import numpy as np



def ranking(head, relation, tail, head_t_w, relation_t_w, tail_t_w, ent_num, kge_model):
    """
    This function computes ranking and hits10 value for the trained model

    @Parameter:
    head, relation, tail (np.array()) : head, relation tail numpy indexes from KGVT class
    ent_num (int): total number of words in ALL dataset, i.e. word.dict maximal index +1
    kge_model : name of trained model
    @Returns:
    mean_rank, hits10 (float)
    """
    ranks = torch.ones(len(head)).to(device)
    all_entity = torch.LongTensor(np.arange(0, ent_num)).to(device)
    head_expand = torch.ones(ent_num).to(device)
    tail_expand = torch.ones(ent_num).to(device)
    relation_expand = torch.ones(ent_num).to(device)

    total_rank = 0
    total_reciprocal_rank = 0
    mean_rank = {}
    hits10 = 0
    with torch.no_grad():
        for idx in range(len(head)):
            # expand head, relation, tail
            h, r, t = head[idx] * head_expand, relation[idx] * relation_expand, tail[idx] * tail_expand
            h, r, t = h.type(torch.LongTensor).to(device), r.type(torch.LongTensor).to(device), t.type(torch.LongTensor).to(device)

            # expand head_word, relation_word, tail_word
            h_w, r_w, t_w = [head_t_w[idx]] * ent_num, [relation_t_w[idx]] * ent_num, [tail_t_w[idx]] * ent_num

            # Replace tail entity with all entities
            Corrupted_score_tail = kge_model.score(h, r, all_entity, h_w, r_w, t_w, 'training')

            # Sort tail entities
            argsort_tail = torch.argsort(Corrupted_score_tail, dim=0, descending=True)

            # get the first([0]) correctly predicted tail ranking index
            ranking_tail = (argsort_tail == t).nonzero(as_tuple=True)[0]
            avg_rank = ranking_tail
            total_rank = total_rank + avg_rank

            # MRR
            reciprocal_rank = 1 / (ranking_tail + 1)
            total_reciprocal_rank += reciprocal_rank
            # hit@10
            hits10 += 1 if avg_rank < 11 else 0
            if idx % 50 == 0:
                print(idx, len(head), hits10)


        mean_rank = total_rank / len(head)
        hits10 = hits10 / len(head)
        mrr = total_reciprocal_rank / len(head)

        print(f'mean_rank:{mean_rank}\thits10:{hits10}\tmrr:{mrr}')

        return mean_rank, hits10, mrr
