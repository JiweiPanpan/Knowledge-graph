###
# traverse .txt file and replace all words with index read from word.dict
###

import pickle as pkl
import numpy as np
import pickle
import os

def replace_words_with_index(text, word_dict):
    words = text.split()
    indexed_words = [word_dict.get(word, -1) for word in words]
    return indexed_words

if __name__ == '__main__':
    with open('word.dict', 'rb') as file:
        word_dict2idx = pickle.load(file)
    file = ['train', 'test', 'valid']
    for filename in file:

        data = []
        with open(filename + '.txt', 'r', encoding='utf-8') as file:
            for line in file:
                triple = line.strip().split('\t')
                replaced_triple = [replace_words_with_index(entity, word_dict2idx) for entity in triple]
                #print(replaced_triple)
                data.append(replaced_triple)



        # print(data)
        #print(np.array(data[0][0]))

        save_path = filename + 'word2idx.txt' + '.pickle'
        with open(save_path, 'wb') as save_file:
            pkl.dump(data, save_file)

        with open(save_path, 'rb') as infile:
            word2idx_list = pkl.load(infile)
            print(word2idx_list[0][0])
            print(word2idx_list)
