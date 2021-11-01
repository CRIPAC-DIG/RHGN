import numpy as np
import pandas as pd
import tqdm
import dgl
import time
import torch
from torch import nn
from utils import *
from utils import neighbormap,split_char,filter_sample,combination
from parser import get_args

import fasttext
np.random.seed(20)
args=get_args()



class dataloader():
    def __init__(self,data_dir=None,):
        # df_user = pd.read_csv('{}/user_profile.csv'.format(data_dir))

        # df_item = pd.read_csv('{}/ad_feature.csv'.format(data_dir))
        # print(df_item.info(verbose=True, null_counts=True))
        # df_item.dropna(axis=0, subset=['cate_id', 'campaign_id', 'brand'], inplace=True)  # 删除必要属性为空的样本
        #
        # print(df_item.info(verbose=True, null_counts=True))
        # df_item.reset_index(drop=True,inplace=True)
        #
        #
        # df_click = pd.read_csv('{}/raw_sample.csv'.format(data_dir))
        # df_click.drop_duplicates(subset=['user', 'adgroup_id'], inplace=True, ignore_index=True)    #删除重复点击样本可以减少图中重复的边
        #
        # #删除没交互的行为
        # df_click=df_click[df_click.clk == 1]
        # df_click.reset_index(drop=True,inplace=True)
        #
        # print('df_click:', df_click.shape)
        #
        # users = set(df_click.user.tolist())
        # items = set(df_click.adgroup_id.tolist())
        # df_user= df_user[df_user['userid'].isin(users)]
        # df_item= df_item[df_item['adgroup_id'].isin(items)]
        # df_user.reset_index(drop=True, inplace=True)
        # df_item.reset_index(drop=True, inplace=True)
        #
        # df_user.rename(columns={'userid': 'user_id'}, inplace=True)
        # df_click.rename(columns={'user': 'user_id'}, inplace=True)
        # df_user.to_csv('{}/user'.format(data_dir), index=False)
        # df_item.to_csv('{}/item_info'.format(data_dir), index=False)
        # df_click.to_csv('{}/click'.format(data_dir), index=False)
        # exit()
        df_user=pd.read_csv('{}/user'.format(data_dir))
        df_item=pd.read_csv('{}/item_info'.format(data_dir))
        df_click=pd.read_csv('{}/click'.format(data_dir))
        df_user = df_user.astype({'user_id': 'str'}, copy=False)
        df_item = df_item.astype({'adgroup_id': 'str', 'cate_id': 'str', 'campaign_id': 'str', 'brand': 'str'},
                                 copy=False)
        df_click = df_click.astype({'user_id': 'str', 'adgroup_id': 'str'}, copy=False)

        self.user_dic = {k: v for v, k in enumerate(df_user.user_id)}
        self.cate_dic = {k: v for v, k in enumerate(df_item.cate_id.drop_duplicates())}  # 构建字典，删除重复项
        self.campaign_dic = {k: v for v, k in enumerate(df_item.campaign_id.drop_duplicates())}
        self.brand_dic = {k: v for v, k in enumerate(df_item.brand.drop_duplicates())}
        self.item_dic = {}
        self.c1,self.c2,self.c3=[],[],[]
        for i in range(len(df_item)):
            k=df_item.at[i,'adgroup_id']
            v=i
            self.item_dic[k]=v
            self.c1.append(self.cate_dic[df_item.at[i,'cate_id']])
            self.c2.append(self.campaign_dic[df_item.at[i,'campaign_id']])

            self.c3.append(self.brand_dic[df_item.at[i,'brand']])
        print(min(self.c1),min(self.c2),min(self.c3))
        print(len(self.cate_dic),len(self.campaign_dic),len(self.brand_dic))
        df_click=df_click[df_click['adgroup_id'].isin(self.item_dic)]
        df_click=df_click[df_click['user_id'].isin(self.user_dic)]
        df_click.reset_index(drop=True, inplace=True)

        self.df_user=df_user
        self.df_item=df_item
        self.df_click=df_click
        self.data_dir=data_dir


    def generate_pure_graph(self,is_save=True):


        click_user = [self.user_dic[user] for user in self.df_click.user_id ]
        click_item = [self.item_dic[item] for item in self.df_click.adgroup_id ]
        data_dict = {
            ('user', 'click', 'item'): (torch.tensor(click_user), torch.tensor(click_item)),
            ('item', 'click-by', 'user'): (torch.tensor(click_item), torch.tensor(click_user)),
                    }
        G = dgl.heterograph(data_dict)


        model = fasttext.load_model('../data/fasttext/fastText/cc.zh.200.bin')
        temp = {k: model.get_sentence_vector(v) for v, k in self.cate_dic.items()}
        cid1_feature = torch.tensor([temp[k] for _, k in self.cate_dic.items()])

        temp = {k: model.get_sentence_vector(v) for v, k in self.campaign_dic.items()}
        cid2_feature = torch.tensor([temp[k] for _, k in self.campaign_dic.items()])

        temp = {k: model.get_sentence_vector(v) for v, k in self.brand_dic.items()}
        cid3_feature = torch.tensor([temp[k] for _, k in self.brand_dic.items()])


        '''将标签传进label'''
        label_gender = self.df_user.final_gender_code
        label_age = self.df_user.age_level
        G.nodes['user'].data['gender'] = torch.tensor(label_gender[:G.number_of_nodes('user')])
        G.nodes['user'].data['age'] = torch.tensor(label_age[:G.number_of_nodes('user')])
        G.nodes['item'].data['cid1'] = torch.tensor(self.c1[:G.number_of_nodes('item')])
        G.nodes['item'].data['cid2'] = torch.tensor(self.c2[:G.number_of_nodes('item')])
        G.nodes['item'].data['cid3'] = torch.tensor(self.c3[:G.number_of_nodes('item')])
        print(G.nodes['item'].data['cid1'].shape,)
        print(G.nodes['item'].data['cid2'].shape)
        print(G.nodes['item'].data['cid3'].shape)
        print(G)
        print(cid1_feature.shape,)
        print(cid2_feature.shape,)
        print(cid3_feature.shape,)
        exit()
        if is_save == True:
            torch.save(G, '{}/G_ori.pkl'.format(self.data_dir))
            torch.save(cid1_feature, '{}/cid1_feature.npy'.format(self.data_dir))
            torch.save(cid2_feature, '{}/cid2_feature.npy'.format(self.data_dir))
            torch.save(cid3_feature, '{}/cid3_feature.npy'.format(self.data_dir))
        self.G = G
        return G


    def generate_homo_graph(self,is_save=True):
        # 获得item的属性
        t = time.time()


        items=self.df_item.adgroup_id.tolist()
        new_item_dic={k:v for v,k in enumerate(items)}
        print('时间1：', time.time() - t)


        init_dic = {self.user_dic[user]: [] for user in self.df_user.user_id}
        # user_item_dic = neighbormap(self.df_order,init_dic , self.user_dic,new_item_dic)  # {user1:[item1,item2...],...}
        user_item_dic = neighbormap(self.df_click,init_dic , self.user_dic,new_item_dic,col_item='adgroup_id')
        user_item, del_user = filter_sample(1, user_item_dic)
        print('时间2：', time.time() - t)

        new_df_user=self.df_user.drop(del_user)
        new_df_user.reset_index(drop=True,inplace=True)
        users=new_df_user.user_id.tolist()
        new_user_dic={k:v for v,k in enumerate(users)}
        print('删除的user数：{}'.format(len(del_user)))
        print('用户数：',len(user_item))


        # co_purchase_user = combination(self.df_order,users)
        # co_purchase_user[0] = [new_user_dic[user] for user in co_purchase_user[0]]
        # co_purchase_user[1] = [new_user_dic[user] for user in co_purchase_user[1]]

        co_click_user = combination(self.df_click,users,col_item='adgroup_id')
        co_click_user[0] = [new_user_dic[user] for user in co_click_user[0]]
        co_click_user[1] = [new_user_dic[user] for user in co_click_user[1]]

        data_dict={
            # ('user', 'co-purchase', 'user'): (torch.tensor(co_purchase_user[0]), torch.tensor(co_purchase_user[1])),
            # ('user', 'co-purchase-by', 'user'): (torch.tensor(co_purchase_user[1]), torch.tensor(co_purchase_user[0])),
            ('user', 'co-click', 'user'): (torch.tensor(co_click_user[0]), torch.tensor(co_click_user[1])),
            # ('user', 'co-click-by', 'user'): (torch.tensor(co_click_user[1]), torch.tensor(co_click_user[0])),
        }
        G = dgl.heterograph(data_dict)
        print('时间3：', time.time() - t)
        # label
        label_gender = new_df_user.final_gender_code.tolist()
        label_age =  new_df_user.age_level.tolist()
        G.nodes['user'].data['gender'] = torch.tensor(label_gender[:G.number_of_nodes('user')])
        G.nodes['user'].data['age'] = torch.tensor(label_age[:G.number_of_nodes('user')])
        total_items=len(items)
        G.nodes['user'].data['all_word'] = torch.randint(low=0,high=total_items,size=(G.number_of_nodes('user'),10,10))
        G = dgl.to_homogeneous(G,ndata=['gender','age','all_word'])
        print('时间4：', time.time() - t)

        word_feature = nn.Parameter(torch.Tensor(total_items, args.n_inp), requires_grad = False)
        nn.init.xavier_uniform_(word_feature)
        if is_save:
            torch.save(G,'{}/homo_G.pkl'.format(self.data_dir))
            torch.save(word_feature, '{}/word_feature.npy'.format(self.data_dir))
        return G




if __name__ == '__main__':
    data_dir='../taobao_data/'
    print('文件路径：',data_dir)
    data=dataloader(data_dir=data_dir)
    G=data.generate_pure_graph(is_save=True)
    # G=data.generate_homo_graph(is_save=True)
    print(G)

