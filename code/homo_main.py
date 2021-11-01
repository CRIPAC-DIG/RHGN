import scipy.io
import dgl
import math
import os
import torch
import numpy as np
from model import *
from baseline import *
from homo_parser import get_args
from sklearn.metrics import f1_score

args=get_args()
'''固定随机种子'''
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda",args.gpu)
# device = torch.device("cpu")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def Batch_train(model, G):
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
            Batch_logits,Batch_labels = model(blocks,label_key=args.label)
            # The loss is computed only for labeled nodes.
            loss = F.cross_entropy(Batch_logits, Batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            train_step += 1
            scheduler.step(train_step)

            acc = torch.sum(Batch_logits.argmax(1) == Batch_labels).item()
            total_loss += loss.item() * len(output_nodes.cpu())
            total_acc += acc
            count += len(output_nodes.cpu())
            
        train_loss, train_acc = total_loss / count, total_acc / count
        torch.cuda.empty_cache()
        if epoch % 1 == 0:
            model.eval()
            '''-------------------------val-----------------------'''
            total_loss = 0
            total_acc = 0
            count = 0
            preds=[]
            labels=[]
            for input_nodes, output_nodes, blocks in val_dataloader:
                Batch_logits,Batch_labels = model(blocks,label_key=args.label)
                loss = F.cross_entropy(Batch_logits, Batch_labels)
                acc   = torch.sum(Batch_logits.argmax(1)==Batch_labels).item()
                preds.extend(Batch_logits.argmax(1).tolist())
                labels.extend(Batch_labels.tolist())
                total_loss += loss.item() * len(output_nodes.cpu())
                total_acc +=acc
                count += len(output_nodes.cpu())
                
            val_f1 = f1_score(preds, labels, average='macro')
            val_loss,val_acc   = total_loss / count, total_acc / count
            '''------------------------test----------------------'''
            total_loss = 0
            total_acc = 0
            count = 0
            preds=[]
            labels=[]
            for input_nodes, output_nodes, blocks in test_dataloader:
                Batch_logits,Batch_labels = model(blocks,label_key=args.label)
                loss = F.cross_entropy(Batch_logits, Batch_labels)
                acc   = torch.sum(Batch_logits.argmax(1)==Batch_labels).item()
                preds.extend(Batch_logits.argmax(1).tolist())
                labels.extend(Batch_labels.tolist())
                total_loss += loss.item() * len(output_nodes.cpu())
                total_acc +=acc
                count += len(output_nodes.cpu())
                
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
            print('\t\tval_f1 %.4f test_f1 \033[1;33m %.4f \033[0m' % (val_f1,test_f1))
        torch.cuda.empty_cache()


'''加载同质图和标签'''
G=torch.load('{}/{}.pkl'.format(args.data_dir,args.graph))
G = dgl.add_self_loop(G)
print(G)
labels=G.ndata[args.label]


# generate train/val/test split
pid = np.arange(len(labels))
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:int(len(labels)*0.8)]).long()
val_idx = torch.tensor(shuffle[int(len(labels)*0.8):int(len(labels)*0.9)]).long()
test_idx = torch.tensor(shuffle[int(len(labels)*0.9):]).long()

print('训练集:',train_idx.shape)
print('验证集:',val_idx.shape)
print('测试集:',test_idx.shape)


G = G.to(device)

'''采样'''
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
# sampler = dgl.dataloading.MultiLayerNeighborSampler([50,20])
train_dataloader = dgl.dataloading.NodeDataLoader(
    G, train_idx.to(device), sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    device=device)

val_dataloader = dgl.dataloading.NodeDataLoader(
    G, val_idx.to(device), sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    device=device)

test_dataloader = dgl.dataloading.NodeDataLoader(
    G, test_idx.to(device), sampler,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=False,
    device=device)



'''同质图'''
if args.model=='GCN':
    word_feature = torch.load('{}/word_feature.npy'.format(args.data_dir))
    model=GCN(in_size=args.n_inp,
                       hidden_size=args.n_hid,
                       out_size=labels.max().item()+1,
                       word_feature=word_feature,).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=args.n_epoch,
                                                    steps_per_epoch=int(train_idx.shape[0]/args.batch_size)+1,max_lr = args.max_lr)
    print('Training GCN with #param: %d' % (get_n_params(model)))
    Batch_train(model, G)

if args.model == 'HGCN':
    word_feature = torch.load('{}/word_feature.npy'.format(args.data_dir))
    print(word_feature.shape)
    model=HGCN(in_size=args.n_inp,
                       hidden_size=args.n_hid,
                       out_size=labels.max().item()+1,
                       item_dim=200,
                       user_dim=200,
                       word_feature=word_feature,
               ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=args.n_epoch,
                                                    steps_per_epoch=int(train_idx.shape[0]/args.batch_size)+1,max_lr = args.max_lr)
    print('Training HGCN with #param: %d' % (get_n_params(model)))
    Batch_train(model, G)

if args.model == 'GAT':
    word_feature = torch.load('{}/word_feature.npy'.format(args.data_dir))
    model=GAT(in_size=args.n_inp,
              hidden_size=args.n_hid,
              out_size=labels.max().item()+1,
              heads=([8] * 1) + [1],  # [8,1]
              activation=torch.nn.PReLU(),
              feat_drop=0.8,
              attn_drop=0.5,
              negative_slope=0.2,
              residual=True,
              word_feature=word_feature,
              ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=args.n_epoch,
                                                    steps_per_epoch=int(train_idx.shape[0]/args.batch_size)+1,max_lr = args.max_lr)
    print('Training GAT with #param: %d' % (get_n_params(model)))
    Batch_train(model, G)

if args.model == 'HGAT':
    word_feature = torch.load('{}/word_feature.npy'.format(args.data_dir))
    model=HGAT(in_size=args.n_inp,
              hidden_size=args.n_hid,
              out_size=labels.max().item()+1,
              item_dim=200,
              user_dim=200,
              heads=([8] * 1) + [1],  # [8,1]
              activation=torch.nn.PReLU(),
              feat_drop=0.8,
              attn_drop=0.5,
              negative_slope=0.2,
              residual=True,
              word_feature=word_feature,
              ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, epochs=args.n_epoch,
                                                    steps_per_epoch=int(train_idx.shape[0]/args.batch_size)+1,max_lr = args.max_lr)
    print('Training HGAT with #param: %d' % (get_n_params(model)))
    Batch_train(model, G)
