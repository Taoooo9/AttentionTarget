import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Bilstm(nn.Module):

    def __init__(self, config, embedding, pad):
        super(Bilstm, self).__init__()
        self.embedding = nn.Embedding(config.embed_num, config.embed_dim, padding_idx=pad,
                                      max_norm=config.max_norm)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        self.lstm = nn.LSTM(config.embed_dim, config.hidden_size, dropout=config.dropout_rnn, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout_embed)

    def forward(self, x, local, add_num, max_add, length):
        left = []
        tar = []
        right = []
        x = self.embedding(x)
        x = self.dropout(x)
        x = pack_padded_sequence(x, length, batch_first=True)
        x, _ = self.lstm(x)
        x = x.data
        start = 0
        for idx, i in enumerate(length):
            add_left = torch.zeros((add_num[idx][0], 300), dtype=torch.float, requires_grad=False)
            add_tar = torch.zeros((add_num[idx][1], 300), dtype=torch.float, requires_grad=False)
            add_right = torch.zeros((add_num[idx][2], 300), dtype=torch.float, requires_grad=False)
            sen_tensor = x[start:i+start]
            if add_left.size(0) == 0:
                left_ten = sen_tensor[0:int(local[idx][0])]
            else:
                left_ten = torch.cat((sen_tensor[0:int(local[idx][0])], add_left), 0)
            if add_tar.size(0) == 0:
                tar_ten = sen_tensor[int(local[idx][0]):int(local[idx][1] + 1)]
            else:
                tar_ten = torch.cat((sen_tensor[int(local[idx][0]):int(local[idx][1] + 1)], add_tar), 0)
            if add_right.size(0) == 0:
                right_ten = sen_tensor[int(local[idx][1]) + 1:]
            else:
                right_ten = torch.cat((sen_tensor[int(local[idx][1]) + 1:], add_right), 0)


            if left_ten.size() == torch.Size([0]):
                left_ten = torch.zeros((max_add[0], 300), dtype=torch.float)
            else:
                left_ten = left_ten.unsqueeze(0)
            if tar_ten.size() == torch.Size([0]):
                tar_ten = torch.zeros((max_add[1], 300), dtype=torch.float)
            else:
                tar_ten = tar_ten.unsqueeze(0)
            if right_ten.size() == torch.Size([0]):
                right_ten = torch.zeros((max_add[2], 300), dtype=torch.float)
            else:
                right_ten = right_ten.unsqueeze(0)
            left.append(left_ten)
            tar.append(tar_ten)
            right.append(right_ten)
            start = i
        left_matrix = torch.cat(left, 0)
        tar_matrix = torch.cat(tar, 0)
        right_matrix = torch.cat(right, 0)
        return left_matrix, tar_matrix, right_matrix

