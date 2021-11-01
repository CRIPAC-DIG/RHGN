import numpy as np
import pandas as pd
import tqdm
import dgl
import torch
from torch import nn
from utils import *
np.random.seed(20)

def delnode(G,etypes,therehold=None):
    index = []
    for etype in etypes:
        temp=[]
        for node,degree in enumerate(G.out_degrees(etype=etype)):
            if degree>1 :
                pass
            else:
                temp.append(node)
        index.append(temp)
    index=[i for i in index[0] if i in index[1]]
    return index



df_user=pd.read_csv('../data/user')
df_user=df_user.astype({'user_id':'str'},copy=False)
user_dic={k:v for v,k in enumerate(df_user.user_id)}
age_dic={k:v for v,k in enumerate(df_user.age_range.drop_duplicates())}
#label
label_gender = df_user.gender
label_age = [age_dic[age] for age in df_user.age_range]


df_item=pd.read_csv('../data/item_info')
df_item=df_item.astype({'item_id': 'str','cid1':'str',
                        'cid2':'str','cid3':'str','brand_code':'str'},copy=False)
cid1_dic={k:v for v,k in enumerate(df_item.cid1.drop_duplicates())}     #构建字典，删除重复项
cid2_dic={k:v for v,k in enumerate(df_item.cid2.drop_duplicates())}
cid3_dic={k:v for v,k in enumerate(df_item.cid3.drop_duplicates())}
brand_dic={k:v for v,k in enumerate(df_item.brand_code.drop_duplicates())}
item_dic={k:v for v,k in enumerate(df_item.item_id)}

df_click=pd.read_csv('../data/user_click')
df_click=df_click.astype({'user_id':'str','item_id': 'str'},copy=False)
index=[]        #删除item在样本外的点击记录
for i,item in enumerate(df_click.item_id):
    if item not in item_dic.keys():
        index.append(i)
df_click.drop(index,inplace=True)
print('df_click:',df_click.shape[0])

df_order=pd.read_csv('../data/user_order')
df_order=df_order.astype({'user_id':'str','item_id': 'str'},copy=False)
index1=[]       #删除item在样本外的购买记录
for i,item in enumerate(df_order.item_id):
    if item not in item_dic.keys():
        index1.append(i)
df_order.drop(index1,inplace=True)
print('df_order:',df_order.shape[0])

item=[item_dic[item] for item in df_item.item_id]
cid1=[cid1_dic[cid] for cid in df_item.cid1]
cid2=[cid2_dic[cid] for cid in df_item.cid2]
cid3=[cid3_dic[cid] for cid in df_item.cid3]
brand=[brand_dic[brand] for brand in df_item.brand_code]
click_user=[user_dic[user] for user in df_click.user_id]
click_item=[item_dic[item] for item in df_click.item_id]
purchase_user=[user_dic[user] for user in df_order.user_id]
purchase_item=[item_dic[item] for item in df_order.item_id]

data_dict = {
    ('user', 'click', 'item'): (torch.tensor(click_user), torch.tensor(click_item)),
    ('item', 'click-by', 'user'): (torch.tensor(click_item), torch.tensor(click_user)),
    ('user', 'purchase', 'item'): (torch.tensor(purchase_user), torch.tensor(purchase_item)),
    ('item', 'purchase-by', 'user'): (torch.tensor(purchase_item), torch.tensor(purchase_user)),
    # ('cid1', 'relation1_cid', 'item'): (torch.tensor(cid1), torch.tensor(item)),
    # ('cid2', 'relation2_cid', 'item'): (torch.tensor(cid2), torch.tensor(item)),
    # ('cid3', 'relation3_cid', 'item'): (torch.tensor(cid3), torch.tensor(item)),
    # ('brand', 'include', 'item'): (torch.tensor(brand), torch.tensor(item)),
    # ('item', 'include-by', 'brand'): (torch.tensor(item), torch.tensor(brand)),
}
G = dgl.heterograph(data_dict)
'''将标签传进label'''
G.nodes['user'].data['gender'] = torch.tensor(label_gender[:G.nodes('user').shape[0]])
G.nodes['user'].data['age'] =   torch.tensor(label_age[:G.nodes('user').shape[0]])


'''删除度<1的节点'''
# index=delnode(G,['click','purchase'],therehold=5)
# index1=delnode(G,['click-by','purchase-by'],therehold=5)
# G.remove_nodes(index,ntype='user')
# G.remove_nodes(index1,ntype='item')

print(G)
torch.save(G,'../data/G.pkl')



