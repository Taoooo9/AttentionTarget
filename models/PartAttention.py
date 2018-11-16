import torch
import torch.nn as nn
import torch.nn.functional as F


class PartAttention(nn.Module):
    def __init__(self, args):
        super(PartAttention, self).__init__()
        self.args = args
        self.w1 = nn.Linear(args.embed_dim * 2, args.attention_size, bias=True)
        nn.init.xavier_uniform_(self.w1.weight)
        self.u = nn.Linear(args.attention_size, 1, bias=True)
        nn.init.xavier_uniform_(self.u.weight)


    def forward(self, part_matrix, tar_matrix):
        ht = torch.mean(tar_matrix, 1)
        ht = ht.unsqueeze(1)
        ht = ht.repeat(1, part_matrix.size(1), 1)
        h = torch.cat((part_matrix, ht), 2)
        h = self.w1(h)
        h = F.tanh(h)
        beta = self.u(h)
        alpha = F.softmax(beta, 0)
        alpha = alpha.repeat(1, 1, part_matrix.size(2))
        s = torch.mul(alpha, part_matrix)
        s = torch.sum(s, 1)
        return s
