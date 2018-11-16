import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args
        self.w1 = nn.Linear(args.embed_dim * 2, args.attention_size, bias=True)
        nn.init.xavier_uniform_(self.w1.weight)
        self.u = nn.Linear(args.attention_size, 1, bias=True)
        nn.init.xavier_uniform_(self.u.weight)


    def forward(self, left_matrix, tar_matrix, right_matrix):
        ht = torch.mean(tar_matrix, 1)
        ht = ht.unsqueeze(1)
        hi = torch.cat((left_matrix, right_matrix), 1)
        ht = ht.repeat(1, hi.size(1), 1)
        h = torch.cat((hi, ht), 2)
        h = self.w1(h)
        h = F.tanh(h)
        beta = self.u(h)
        alpha = F.softmax(beta, 0)
        alpha = alpha.repeat(1, 1, hi.size(2))
        s = torch.mul(alpha, hi)
        s = torch.sum(s, 1)
        return s











