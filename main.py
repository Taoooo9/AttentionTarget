import argparse
import numpy as np
import random
import os
import pickle
from configer.Config import Configer
from dataloader import analyse_data, dataloader, Vocab
from models.Vanilla import Vanilla
from models.Contextualized import Contextualized
from models.ContextualizedGate import ContextualizedGate
from train.train import train

if __name__ == '__main__':

    random.seed(666)
    np.random.seed(666)

    # 分析数据
    file_train = './data/Z_data/all.conll.train'
    file_dev = './data/Z_data/all.conll.dev'
    file_test = './data/Z_data/all.conll.test'
    analyse_data.analyse(file_train)
    analyse_data.analyse(file_dev)
    analyse_data.analyse(file_test)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='default.ini')
    parser.add_argument('--theard_num', type=int, default=1)
    args, extra_args = parser.parse_known_args()
    config = Configer(args.config_file, extra_args)

    # dataloder
    src_tra_word, label_tra_word = dataloader.read_data(file_train)
    src_dev_word, label_dev_word = dataloader.read_data(file_dev)
    src_test_word, label_test_word = dataloader.read_data(file_test)
    if not os.path.isdir(config.save_dir):
        os.mkdir(config.save_dir)
    pickle.dump(src_tra_word, open(config.train_pkl, 'wb'))
    if config.dev_file:
        pickle.dump(src_dev_word, open(config.dev_pkl, 'wb'))
    pickle.dump(src_test_word, open(config.test_pkl, 'wb'))

    # vocab
    src_vocab = Vocab.VocabSrc(src_tra_word, config)
    tar_vocab = Vocab.VocabLab(label_tra_word, config)

    # embedding
    embedding = None
    if config.embedding_pkl:
        embedding = src_vocab.create_embedding()
        pickle.dump(embedding, open(config.embedding_pkl, mode='wb'))

    # model
    if config.which_model == 'Vanilla':
        model = Vanilla(config, embedding, Vocab.PAD)
    elif config.which_model == 'Contextualized':
        model = Contextualized(config, embedding, Vocab.PAD)
    elif config.which_model == 'ContextualizedGate':
        model = ContextualizedGate(config, embedding, Vocab.PAD)
    else:
        raise RuntimeError('Please chooice ture model!')

    # train
    train(model, src_tra_word, src_dev_word, src_test_word, src_vocab, tar_vocab, config)









