import torch
import torch.nn as nn
from .Attention import Attention
from models.BiLstm import Bilstm
import torch.nn.functional as F


class Vanilla(nn.Module):

    def __init__(self, args, embedding, pad):
        super(Vanilla, self).__init__()
        self.lstm = Bilstm(args, embedding, pad)
        self.attention = Attention(args)
        self.w = nn.Linear(args.embed_dim, args.class_num, bias=True)
        torch.nn.init.xavier_uniform_(self.w.weight)


    def forward(self, x, local, add_num, max_add, length):
        left_matrix, tar_matrix, right_matrix = self.lstm(x, local, add_num, max_add, length)
        x = self.attention(left_matrix, tar_matrix, right_matrix)
        logit = F.softmax(self.w(x), 0)
        return logit







