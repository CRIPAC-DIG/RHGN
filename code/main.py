import scipy.io
import dgl
import math
import torch
import numpy as np
from model import *
from utils import *
import argparse
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='Training GNN on ogbn-products benchmark')

parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--n_hid',   type=int, default=32)
parser.add_argument('--n_inp',   type=int, default=200)
parser.add_argument('--clip',    type=int, default=1.0)
parser.add_argument('--max_lr',  type=float, default=1e-2)
parser.add_argument('--label',  type=str, default='gender',choices=['age','gender'])
parser.add_argument('--gpu',  type=int, default=0,choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--graph',  type=str, default='G_ori')
parser.add_argument('--model',  type=str, default='HGT',choices=['HGT','RGCN'])
parser.add_argument('--data_dir',  type=str, default='../data/sample')

args = parser.parse_args()
'''固定随机种子'''
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def Batch_train(model):
    best_val_acc = 0
    best_test_acc = 0
    train_step = 0
    Minloss_val = 10000.0
    for epoch in np.arange(args.n_epoch) + 1:
        model.train()
        '''---------------------------train------------------------'''
        total_loss = 0
        total_acc = 0
        count = 0
        for input_nodes, output_nodes, blocks in train_dataloader:
            Batch_logits,Batch_labels = model(input_nodes,output_nodes,blocks, out_key='user',label_key=args.label, is_train=True)

            # The loss is computed only for labeled nodes.
            loss = F.cross_entropy(Batch_logits, Batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_step += 1
            scheduler.step(train_step)

            acc = torch.sum(Batch_logits.argmax(1) == Batch_labels).item()
            total_loss += loss.item() * len(output_nodes['user'].cpu())
            total_acc += acc
            count += len(output_nodes['user'].cpu())
        train_loss, train_acc = total_loss / count, total_acc / count

        if epoch % 1 == 0:
            model.eval()
            '''-------------------------val-----------------------'''
            total_loss = 0
            total_acc = 0
            count = 0
            preds=[]
            labels=[]
            for input_nodes, output_nodes, blocks in val_dataloader:
                Batch_logits,Batch_labels = model(input_nodes, output_nodes,blocks, out_key='user',label_key=args.label, is_train=False)
                loss = F.cross_entropy(Batch_logits, Batch_labels)
                acc   = torch.sum(Batch_logits.argmax(1)==Batch_labels).item()
                preds.extend(Batch_logits.argmax(1).tolist())
                labels.extend(Batch_labels.tolist())
                total_loss += loss.item() * len(output_nodes['user'].cpu())
                total_acc +=acc
                count += len(output_nodes['user'].cpu())

            val_f1 = f1_score(preds, labels, average='macro')
            val_loss,val_acc   = total_loss / count, total_acc / count
            '''------------------------test----------------------'''
            total_loss = 0
            total_acc = 0
            count = 0
            preds=[]
            labels=[]
            for input_nodes, output_nodes, blocks in test_dataloader:
                Batch_logits,Batch_labels = model(input_nodes, output_nodes,blocks, out_key='user',label_key=args.label, is_train=False)
                loss = F.cross_entropy(Batch_logits, Batch_labels)
                acc   = torch.sum(Batch_logits.argmax(1)==Batch_labels).item()
                preds.extend(Batch_logits.argmax(1).tolist())
                labels.extend(Batch_labels.tolist())
                total_loss += loss.item() * len(output_nodes['user'].cpu())
                total_acc +=acc
                count += len(output_nodes['user'].cpu())

            test_f1 = f1_score(preds,labels, average='macro')
            test_loss,test_acc   = total_loss / count, total_acc / count
            if  val_acc   > best_val_acc:
                Minloss_val = val_loss
                best_val_acc = val_acc
                best_test_acc = test_acc
            print('Epoch: %d LR: %.5f Loss %.4f, val loss %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                epoch,
                optimizer.param_groups[0]['lr'],
                train_loss,
                val_loss,
                val_acc,
                best_val_acc,
                test_acc,
                best_test_acc,
            ))
            print('\t\tval_f1 %.4f test_f1 \033[1;33m %.4f \033[0m' % (val_f1, test_f1))
        torch.cuda.empty_cache()

device = torch.device("cuda:{}".format(args.gpu))

'''加载图和标签'''
G=torch.load('{}/{}.pkl'.format(args.data_dir,args.graph))
labels=G.nodes['user'].data[args.label]


# generate train/val/test split
pid = np.arange(len(labels))
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:int(len(labels)*0.75)]).long()
val_idx = torch.tensor(shuffle[int(len(labels)*0.75):int(len(labels)*0.875)]).long()
test_idx = torch.tensor(shuffle[int(len(labels)*0.875):]).long()

print(train_idx.shape)
print(val_idx.shape)
print(test_idx.shape)


node_dict = {}
edge_dict = {}
for ntype in G.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in G.etypes:
    edge_dict[etype] = len(edge_dict)
    G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

#     Initialize input feature
# import fasttext
# model = fasttext.load_model('../data/fasttext/fastText/cc.zh.200.bin')
# sentence_dic=torch.load('../data/sentence_dic.pkl')
# sentence_vec = [model.get_sentence_vector(sentence_dic[k]) for k, v in enumerate(G.nodes('item').tolist())]
# for ntype in G.ntypes:
#     if ntype=='item':
#         emb=nn.Parameter(torch.Tensor(sentence_vec), requires_grad = False)
#     else:
#         emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 200), requires_grad = False)
#         nn.init.xavier_uniform_(emb)
#     G.nodes[ntype].data['inp'] = emb
#
for ntype in G.ntypes:
    emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), args.n_inp), requires_grad = False)
    nn.init.xavier_uniform_(emb)
    G.nodes[ntype].data['inp'] = emb


G = G.to(device)
train_idx_item=torch.tensor(shuffle[0:int(G.number_of_nodes('item') * 0.75)]).long()
val_idx_item = torch.tensor(shuffle[int(G.number_of_nodes('item')*0.75):int(G.number_of_nodes('item')*0.875)]).long()
test_idx_item = torch.tensor(shuffle[int(G.number_of_nodes('item')*0.875):]).long()
'''采样'''
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
train_dataloader = dgl.dataloading.NodeDataLoader(
    G, {'user':train_idx.to(device)}, sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    device=device)

val_dataloader = dgl.dataloading.NodeDataLoader(
    G, {'user':val_idx.to(device)}, sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    device=device)

test_dataloader = dgl.dataloading.NodeDataLoader(
    G, {'user':test_idx.to(device)}, sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    device=device)


if args.model=='HGT':

    model = Batch_HGT(G,
                node_dict, edge_dict,
                n_inp=args.n_inp,
                n_hid=args.n_hid,
                n_out=labels.max().item()+1,
                n_layers=2,
                n_heads=4,
                use_norm = True).to(device)
    optimizer = torch.optim.AdamW(model.parameters())

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=args.n_epoch,
                                                    steps_per_epoch=int(train_idx.shape[0]/args.batch_size)+1,max_lr = args.max_lr)
    print('Training HGT with #param: %d' % (get_n_params(model)))
    Batch_train(model)

if args.model=='RGCN':
    model = HeteroGCN(G,
                       in_size=args.n_inp,
                       hidden_size=args.n_hid,
                       out_size=labels.max().item()+1).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=args.n_epoch,
                                                    steps_per_epoch=int(train_idx.shape[0]/args.batch_size)+1,max_lr = args.max_lr)
    print('Training RGCN with #param: %d' % (get_n_params(model)))
    Batch_train(model)



