import numpy as np
from collections import Counter
import pickle

PAD, UNK = 0, 1
PAD_S, UNK_S = '<pad>', '<unk>'

class VocabSrc(object):

    def __init__(self, src, config):
        self.word2id = {}
        self.config = config
        word_counter = Counter()
        for word_list in src:
            for word in word_list[0]:
                word_counter[word] += 1
        most_word = [k for k, v in word_counter.most_common(config.vocab_size)]
        pickle.dump(most_word, open(config.save_feature_voc, mode='wb'))
        self.id2word = [PAD_S, UNK_S] + most_word
        for idx, word in enumerate(self.id2word):
            self.word2id[word] = idx

    def w2i(self, xx):
        if isinstance(xx, list):
            return [self.word2id.get(word, UNK) for word in xx]
        return self.word2id.get(xx, UNK)

    def i2d(self, xx):
        if isinstance(xx, list):
            return [self.id2word[idx] for idx in xx]
        return self.id2word[xx]

    def create_embedding(self):
        embedding_num = 0
        embedding_dim = 0
        find_count = 0
        embedding = np.zeros((len(self.word2id), 1))
        with open(self.config.embedding_file, encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                line = line.split()
                if embedding_num == 0:
                    embedding_dim = len(line) - 1
                    embedding = np.zeros((len(self.word2id), embedding_dim))
                if line[0] in self.id2word:
                    find_count += 1
                    vector = np.array(line[1:], dtype='float64')
                    embedding[self.word2id[line[0]]] = vector
                    embedding[UNK] += vector
                embedding_num += 1
        not_find = len(self.word2id) - find_count
        oov_ratio = float(not_find / len(self.word2id))
        embedding[UNK] = embedding[UNK] / find_count
        embedding = embedding / np.std(embedding)
        print('\nTotal words: ' + str(embedding_num))
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')
        print('oov ratio: {:.4f}'.format(oov_ratio))
        return embedding

    @property
    def size(self):
        return len(self.id2word)

class VocabLab(object):

    def __init__(self, label, config):
        self.word2id = {}
        label_counter = Counter()
        for tar_list in label:
            label_counter[tar_list[-1]] += 1
        label_list = [k for k, v in label_counter.most_common(config.tar_num)]
        pickle.dump(label_list, open(config.save_label_voc, mode='wb'))
        print(label_list)
        self.id2word = label_list
        for idx, label in enumerate(self.id2word):
            self.word2id[label] = idx

    def w2i(self, xx):
        if isinstance(xx, list):
            return [self.word2id.get(word) for word in xx]
        return self.word2id.get(xx)

    def i2w(self, xx):
        if isinstance(xx, list):
            return [self.id2word[idx] for idx in xx]
        return self.id2word[xx]

    @property
    def size(self):
        return len(self.word2id)




