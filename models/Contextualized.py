import torch
import torch.nn as nn
import torch.nn.functional as F
from .Attention import Attention
from models.BiLstm import Bilstm
from .PartAttention import PartAttention

class Contextualized(nn.Module):

    def __init__(self, config, embedding, pad):
        super(Contextualized, self).__init__()
        self.lstm = Bilstm(config, embedding, pad)
        self.w = nn.Linear(config.embed_dim, config.class_num, bias=True)
        nn.init.xavier_uniform_(self.w.weight)
        self.w_l = nn.Linear(config.embed_dim, config.class_num, bias=True)
        nn.init.xavier_uniform_(self.w_l.weight)
        self.w_r = nn.Linear(config.embed_dim, config.class_num, bias=True)
        nn.init.xavier_uniform_(self.w_l.weight)
        self.att = Attention(config)
        self.att_l = PartAttention(config)
        self.att_r = PartAttention(config)


    def forward(self, x, local, add_num, max_add, length):
        left_matrix, tar_matrix, right_matrix = self.lstm(x, local, add_num, max_add, length)
        x = self.att(left_matrix, tar_matrix, right_matrix)
        x_l = self.att_l(left_matrix, tar_matrix)
        x_r = self.att_r(right_matrix, tar_matrix)
        s = self.w(x)
        s_l = self.w(x_l)
        s_r = self.w(x_r)
        s = s + s_l + s_r
        logit = F.softmax(s, 0)
        return logit
