import re
import torch
from collections import Counter
import numpy as np
import copy

def read_data(file):
    src_word = []
    src = []
    label_word = []
    label = []
    count = 0
    start = -1
    end = -1
    tag = ''
    first = True
    text = open(file, encoding='utf-8').readlines()
    temp = text[0]
    for line in text[1:]:
        if temp is not '\n':
            count += 1
            temp = temp.strip()
            word = re.findall(r'^(\S+)', temp)
            target = re.findall(r'(\S+)$', temp)
            if target[-1] != 'o' and first is True:
                tag = re.split(r'\S-', target[-1])
                tag = tag[-1]
                start = count
                first = False
            if (target[-1] == 'o' and first is False) or (line == '\n' and first is False):
                if line == '\n' and target[-1] != 'o':
                    end = count
                    first = True
                else:
                    end = count - 1
                    first = True
            src.append(word[-1])
            label.append(target[-1])
            temp = line
        elif temp == '\n':
            temp = line
            src_word.append([src, tag, (start-1, end-1), len(src)])
            label_word.append([label, tag])
            count = 0
            src = []
            label = []
    return src_word, label_word

def statistics(file):
    count = Counter()
    for line in file:
        count[line[2]] += 1
    data = [{k: v} for k, v in count.most_common()]
    print(len(data))
    return data

def create_batch(file_data, batch_size, shuffle=False):
    data = copy.deepcopy(file_data)
    if shuffle:
        np.random.shuffle(data)

    max_left = 0
    max_tar = 0
    max_right = 0
    unit = []
    units = []
    count = 0

    for line in data:
        nation = line[2]
        left = int(nation[0])
        if max_left < left:
            max_left = left
        tar = int(nation[1] - nation[0] + 1)
        if max_tar < tar:
            max_tar = tar
        right = int(line[3] - nation[1] - 1)
        if max_right < right:
            max_right = right
        if count < batch_size:
            unit.append(line)
            count += 1
        if count == batch_size:
            last_unit = []
            max_superscript = ()
            for sentence in unit:
                max_superscript = (max_left, max_tar, max_right)
                local = sentence[2]
                add_left = max_left - int(local[0])
                add_tar = max_tar - int(local[1] - local[0] + 1)
                add_right = max_right - int(sentence[3] - local[1] - 1)
                add_unit = (add_left, add_tar, add_right)
                sentence.append(add_unit)
                last_unit.append(sentence)
            units.append([last_unit, max_superscript])
            unit = []
            count = 0
            max_left = 0
            max_tar = 0
            max_right = 0
    if len(unit) > 0:
        last_unit = []
        max_superscript = ()
        for sentence in unit:
            max_superscript = (max_left, max_tar, max_right)
            local = sentence[2]
            add_left = max_left - int(local[0])
            add_tar = max_tar - int(local[1] - local[0] + 1)
            add_right = max_right - int(sentence[3] - local[1] - 1)
            add_unit = (add_left, add_tar, add_right)
            sentence.append(add_unit)
            last_unit.append(sentence)
        units.append([last_unit, max_superscript])

    for mini_batch in units:
        yield mini_batch


def pair_data_variable(batch_data, src_vocab, tar_vocab, config):
    mini_batch = batch_data[0]
    batch_size = len(mini_batch)
    mini_batch = sorted(mini_batch, key=lambda k: k[:][3], reverse=True)
    max_length = max(mini_batch[i][3] for i in range(0, batch_size))
    add_num = []
    local = []
    sentence_list = []
    src_matrix = torch.zeros((batch_size, max_length), dtype=torch.long, requires_grad=False)
    # src_left = torch.zeros((batch_size, max_list[0]), dtype=torch.float, requires_grad=False)
    # src_tar = torch.zeros((batch_size, max_list[1]), dtype=torch.float, requires_grad=False)
    # src_right = torch.zeros((batch_size, max_list[2]), dtype=torch.float, requires_grad=False)
    tar_matrix = torch.zeros((batch_size, 0), dtype=torch.long, requires_grad=False)
    for idx, instance in enumerate(mini_batch):
        sentence = src_vocab.w2i(instance[0])
        local.append(instance[2])
        add_num.append(instance[4])
        sentence_list.append(instance[3])
        for index, word in enumerate(sentence):
            src_matrix[idx][index] = word
        tar_matrix[idx] = tar_vocab.w2i(instance[1])
    sentence_list = sorted(sentence_list, reverse=True)

    return src_matrix, tar_matrix, local, add_num, sentence_list,

if __name__ == '__main__':
    file = '../data/Z_data/all.conll.train'
    a, b = read_data(file)
    c = statistics(a)




