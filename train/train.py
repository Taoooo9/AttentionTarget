import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from dataloader.dataloader import pair_data_variable, create_batch



def train(model, train_data, dev_data, test_data, src_vocab, tar_vocab, config):
    model.train()
   # parameters = filter(lambda p: p.requires_gard, model.parameters())
    if config.learning_algorithm == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.learning_algorithm == 'sdg':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise RuntimeError('Invalid optimizer method: ' + config.learning_algorithm)

    # Get start!
    global_step = 0
    best_acc = 0
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        print('The epoch is :' + str(epoch))
        batch_iter = 0
        batch_num = int(np.ceil(len(train_data) / float(config.batch_size)))
        for batch in create_batch(train_data, config.batch_size, shuffle=True):
            batch_size = len(batch[0])
            max_add = batch[1]
            start_time = time.time()
            feather, target, local, add_num, sentence_len = pair_data_variable(batch, src_vocab, tar_vocab, config)
            optimizer.zero_grad()
            logit = model(feather, local, add_num, max_add, sentence_len)
            loss = F.cross_entropy(logit, target)
            loss_value = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), config.clip_norm)
            optimizer.step()

            correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            accuracy = 100.0 * correct / batch_size
            during_time = float(time.time() - start_time)
            print('Step:{}, Epoch:{}, batch_iter:{}, accuracy:{:.4f}({}/{}), time:{:.2f}, loss:{:.6f}'
                  .format(global_step, epoch, batch_iter, accuracy, correct, batch_size, during_time, loss_value))
            batch_iter += 1
            global_step += 1

            if batch_iter % config.test_interval == 0 or batch_iter == batch_num:
                dev_acc = evaluate(model, dev_data, global_step, src_vocab, tar_vocab, config)
                test_acc = evaluate(model, test_data, global_step, src_vocab, tar_vocab, config)
                if dev_acc > best_acc:
                    print('History acc is {}%, but now is {}%'.format(best_acc, dev_acc))
                    best_acc = dev_acc
                    if os.path.exists(config.save_model_path):
                        pass
                    else:
                        os.makedirs(config.save_model_path)
                    if -1 < config.save_after <= epoch:
                        torch.save(model.state_dict(), os.path.join(config.save_model_path, 'model.' + str(global_step)))
        one_epoch_time = time.time() - epoch_start_time
        print('One epoch using time:{:.2f}'.format(one_epoch_time))


def evaluate(model, dev_data, step, src_vocab, tar_vocab, config):
    model.eval()
    start_time = time.time()
    corrects, size = 0, 0

    for batch in create_batch(dev_data, config.batch_size, shuffle=True):
        batch_size = len(batch[0])
        max_add = batch[1]
        feather, target, local, add_num, sentence_len = pair_data_variable(batch, src_vocab, tar_vocab, config)
        logit = model(feather, local, add_num, max_add, sentence_len)
        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        corrects += correct
        size += batch_size
    accuracy = 100.0 * corrects / size
    during_time = time.time() - start_time
    print("\nevaluate result: ")
    print("Step:{}, accuracy:{:.4f}({}/{}), time:{:.2f}"
          .format(step, accuracy, corrects, size, during_time))
    model.train()
    return accuracy
















