import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import Attention
from models.BiLstm import Bilstm
from .PartAttention import PartAttention

class ContextualizedGate(nn.Module):

    def __init__(self, config, embedding, pad):
        super(ContextualizedGate, self).__init__()
        self.config = config
        self.lstm = Bilstm(config, embedding, pad)
        self.w4 = nn.Linear(config.embed_dim, 3, bias=True)
        nn.init.xavier_uniform_(self.w4.weight)
        self.att = Attention(config)
        self.att_l = PartAttention(config)
        self.att_r = PartAttention(config)

    @staticmethod
    def linear(x, h, config, max_length):
        w = nn.Linear(config.embed_dim, config.embed_dim, bias=True)
        nn.init.xavier_uniform_(w.weight)
        u = nn.Linear(max_length, 1, bias=True)
        h = u(torch.transpose(h, 1, 2)).squeeze(2)
        z = w(x) + h
        return z

    def forward(self, x, local, add_num, max_add, length):
        max_length = max_add[1]
        left_matrix, tar_matrix, right_matrix = self.lstm(x, local, add_num, max_add, length)
        s = self.att(left_matrix, tar_matrix, right_matrix)
        s_l = self.att_l(left_matrix, tar_matrix)
        s_r = self.att_r(right_matrix, tar_matrix)
        z = self.linear(s, tar_matrix, self.config, max_length)
        z_l = self.linear(s_l, tar_matrix, self.config, max_length)
        z_r = self.linear(s_r, tar_matrix, self.config, max_length)
        z_cat = torch.cat((z, z_l, z_r), 1)
        z = F.softmax(z_cat, 0)
        zz = torch.chunk(z, 3, 1)
        ss = torch.mul(zz[0], s) + torch.mul(zz[1], s_l) + torch.mul(zz[2], s_r)
        logit = F.softmax(self.w4(ss), 0)
        return logit
