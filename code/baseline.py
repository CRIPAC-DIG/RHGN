import torch.nn as nn
import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from layers import *


class GCN(nn.Module):
    def __init__(self,in_size, hidden_size, out_size,word_feature):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_size, hidden_size)
        self.conv2 = dgl.nn.GraphConv(hidden_size, out_size)
        self.word_feature = nn.Embedding(word_feature.size(0), word_feature.size(1))
        self.word_feature.weight = nn.Parameter(word_feature)
        self.word_feature.weight.requires_grad = False
    def forward(self, blocks,label_key):

        user_word = blocks[0].srcdata['all_word']  # (batch_size,10,10)
        word_feature = self.word_feature(user_word)  # 将词转化为词向量      #(N,10,10,200)
        item_feature = torch.mean(word_feature, dim=2)
        x = torch.mean(item_feature, dim=1)

        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        x=x[:blocks[-1].number_of_dst_nodes()]
        labels = blocks[-1].dstdata[label_key]
        return x,labels


class HGCN(nn.Module):
    def __init__(self,in_size, hidden_size, out_size ,item_dim, user_dim,word_feature):
        super(HGCN, self).__init__()

        self.conv1 = dgl.nn.GraphConv(in_size, hidden_size)
        self.conv2 = dgl.nn.GraphConv(hidden_size, out_size)

        self.w1 = Parameter(torch.Tensor(word_feature.size(1), item_dim))
        self.bias1 = Parameter(torch.Tensor(item_dim))
        self.attn1 = Parameter(torch.Tensor(item_dim, 1))
        self.w2 = Parameter(torch.Tensor(item_dim, user_dim))
        self.bias2 = Parameter(torch.Tensor(user_dim))
        self.attn2 = Parameter(torch.Tensor(user_dim, 1))
        self.softmax = nn.Softmax(dim=-1)

        self.word_feature = nn.Embedding(word_feature.size(0), word_feature.size(1))
        self.word_feature.weight = nn.Parameter(word_feature)
        self.word_feature.weight.requires_grad = False


        init.xavier_uniform_(self.w1)
        init.constant_(self.bias1, 0)
        init.xavier_uniform_(self.attn1)

        init.xavier_uniform_(self.w2)
        init.constant_(self.bias2, 0)
        init.xavier_uniform_(self.attn2)
        self.norm = nn.InstanceNorm1d(in_size, momentum=0.0, affine=True)

    def forward(self, blocks,label_key):
                                                        # vertices(batch_size,用户节点数)
        # user_item = self.interaction_item[vertices]     # user_item (batch_size,用户节点数,10)
        # user_word = self.interaction_word[user_item]    # user_word(batch_size,用户节点数,10,10)  取出一串词

        user_word=blocks[0].srcdata['all_word']        #(batch_size,10,10)
        word_feature = self.word_feature(user_word)     # 将词转化为词向量      #(N,10,10,200)

      
        word_feature = torch.matmul(word_feature, self.w1) + self.bias1         #(N,10,10,200)
        attn_coef1 = torch.matmul(torch.tanh(word_feature), self.attn1)        #(N,10,10,1)
        attn_coef1 = self.softmax(attn_coef1).transpose(2, 3)                  #(N,10,1,10)
        item_feature = torch.matmul(attn_coef1, word_feature)
        item_feature = item_feature.squeeze(2)

        item_feature = torch.matmul(item_feature, self.w2) + self.bias2
        attn_coef2 = torch.matmul(torch.tanh(item_feature), self.attn2)
        attn_coef2 = self.softmax(attn_coef2).transpose(1, 2)
        x = torch.matmul(attn_coef2, item_feature)
        x = x.squeeze(1)

        x=x.unsqueeze(0)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = x.squeeze(0)

        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        x=x[:blocks[-1].number_of_dst_nodes()]
        labels = blocks[-1].dstdata[label_key]
        return x,labels


class GAT(nn.Module):
    def __init__(self, in_size, hidden_size, out_size,heads,activation,feat_drop, attn_drop, negative_slope, residual,word_feature):
        super(GAT, self).__init__()
        self.gat1 = dgl.nn.GATConv(in_size, hidden_size, heads[0], feat_drop, attn_drop, negative_slope, residual,activation)
        self.gat2 = dgl.nn.GATConv(hidden_size * heads[-2], out_size, heads[-1],feat_drop, attn_drop, negative_slope, residual, None)

        self.word_feature = nn.Embedding(word_feature.size(0), word_feature.size(1))
        self.word_feature.weight = nn.Parameter(word_feature)
        self.word_feature.weight.requires_grad = False
    def forward(self, blocks,label_key):

        user_word = blocks[0].srcdata['all_word']  # (batch_size,10,10)
        word_feature = self.word_feature(user_word)  # 将词转化为词向量      #(N,10,10,200)
        item_feature = torch.mean(word_feature, dim=2)
        x = torch.mean(item_feature, dim=1)

        x = self.gat1(blocks[0], x).flatten(1)
        x = self.gat2(blocks[1], x).mean(1)
        x = x[:blocks[-1].number_of_dst_nodes()]
        labels = blocks[-1].dstdata[label_key]

        return x,labels


class HGAT(nn.Module):
    def __init__(self, in_size, hidden_size, out_size,item_dim,user_dim,heads,activation,feat_drop, attn_drop, negative_slope, residual,word_feature):
        super(HGAT, self).__init__()
        self.gat1 = dgl.nn.GATConv(in_size, hidden_size, heads[0], feat_drop, attn_drop, negative_slope, residual,activation)
        self.gat2 = dgl.nn.GATConv(hidden_size * heads[-2], out_size, heads[-1],feat_drop, attn_drop, negative_slope, residual, None)

        self.w1 = Parameter(torch.Tensor(word_feature.size(1), item_dim))
        self.bias1 = Parameter(torch.Tensor(item_dim))
        self.attn1 = Parameter(torch.Tensor(item_dim, 1))
        self.w2 = Parameter(torch.Tensor(item_dim, user_dim))
        self.bias2 = Parameter(torch.Tensor(user_dim))
        self.attn2 = Parameter(torch.Tensor(user_dim, 1))
        self.softmax = nn.Softmax(dim=-1)

        self.word_feature = nn.Embedding(word_feature.size(0), word_feature.size(1))
        self.word_feature.weight = nn.Parameter(word_feature)
        self.word_feature.weight.requires_grad = False


        init.xavier_uniform_(self.w1)
        init.constant_(self.bias1, 0)
        init.xavier_uniform_(self.attn1)

        init.xavier_uniform_(self.w2)
        init.constant_(self.bias2, 0)
        init.xavier_uniform_(self.attn2)
        self.norm = nn.InstanceNorm1d(in_size, momentum=0.0, affine=True)
    def forward(self, blocks,label_key):

        user_word=blocks[0].srcdata['all_word']        #(batch_size,10,10)
        word_feature = self.word_feature(user_word)     # 将词转化为词向量      #(N,10,10,200)

        word_feature = torch.matmul(word_feature, self.w1) + self.bias1
        attn_coef1 = torch.matmul(torch.tanh(word_feature), self.attn1)        #(N,10,10,1)
        attn_coef1 = self.softmax(attn_coef1).transpose(2, 3)                  #(N,10,1,10)
        item_feature = torch.matmul(attn_coef1, word_feature)
        item_feature = item_feature.squeeze(2)

        item_feature = torch.matmul(item_feature, self.w2) + self.bias2
        attn_coef2 = torch.matmul(torch.tanh(item_feature), self.attn2)
        attn_coef2 = self.softmax(attn_coef2).transpose(1, 2)
        x = torch.matmul(attn_coef2, item_feature)
        x = x.squeeze(1)

        x=x.unsqueeze(0)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = x.squeeze(0)

        x = self.gat1(blocks[0], x).flatten(1)
        x = self.gat2(blocks[1], x).mean(1)
        x = x[:blocks[-1].number_of_dst_nodes()]
        labels = blocks[-1].dstdata[label_key]

        return x,labels
