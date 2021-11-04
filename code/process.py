import numpy as np
import pandas as pd
import tqdm
import dgl
import time
import torch
from torch import nn
from utils import *
from utils import neighbormap,split_char,filter_sample,combination
from homo_parser import get_args

import fasttext
np.random.seed(20)
args=get_args()



class dataloader():
    def __init__(self,data_dir=None,):
        df_user = pd.read_csv('{}/user'.format(data_dir))
        df_user = df_user.astype({'user_id': 'str'}, copy=False)
        self.user_dic = {k: v for v, k in enumerate(df_user.user_id)}
        # self.age_dic = {k: v for v, k in enumerate(df_user.age_range.drop_duplicates())}
        self.age_dic = {'11~15':0,'16~20':0,'21~25':0,'26~30':1,'31~35':1,'36~40':2,'41~45':2,'46~50':3,'51~55':3,'56~60':4,'61~65':4,'66~70':4,'71~':4}


        df_item = pd.read_csv('{}/item_info'.format(data_dir))
        df_item.dropna(axis=0, subset=['item_name', 'cid1_name', 'cid2_name', 'cid3_name'], inplace=True)  # 删除必要属性为空的样本
        df_item = df_item.astype({'item_id': 'str', 'cid1': 'str', 'cid2': 'str', 'cid3': 'str', 'brand_code': 'str'},
                                 copy=False)
        df_item.reset_index(drop=True,inplace=True)



        self.cid1_dic = {k: v for v, k in enumerate(df_item.cid1_name.drop_duplicates())}  # 构建字典，删除重复项
        self.cid2_dic = {k: v for v, k in enumerate(df_item.cid2_name.drop_duplicates())}
        self.cid3_dic = {k: v for v, k in enumerate(df_item.cid3_name.drop_duplicates())}
        self.brand_dic = {k: v for v, k in enumerate(df_item.brand_code.drop_duplicates())}
        self.item_dic = {}
        self.c1,self.c2,self.c3,self.brand=[],[],[],[]
        for i in range(len(df_item)):
            k=df_item.at[i,'item_id']
            v=i
            self.item_dic[k]=v
            self.c1.append(self.cid1_dic[df_item.at[i,'cid1_name']])
            self.c2.append(self.cid2_dic[df_item.at[i,'cid2_name']])
            self.c3.append(self.cid3_dic[df_item.at[i,'cid3_name']])
            self.brand.append(self.brand_dic[df_item.at[i,'brand_code']])

        df_order = pd.read_csv('{}/user_order'.format(data_dir))
        df_order.drop_duplicates(subset=['user_id', 'item_id'], inplace=True, ignore_index=True)    #删除重复点击样本可以减少图中重复的边
        df_order = df_order.astype({'user_id': 'str', 'item_id': 'str'}, copy=False)
        index = [i for i,item in enumerate(df_order.item_id)  if item not in self.item_dic.keys()]
        df_order.drop(index, inplace=True) # 如果样本里没有这个item，则删除
        df_order.reset_index(drop=True,inplace=True)
        print('df_order:', df_order.shape)

        df_click = pd.read_csv('{}/user_click'.format(data_dir))
        df_click.drop_duplicates(subset=['user_id', 'item_id'], inplace=True, ignore_index=True)    #删除重复点击样本可以减少图中重复的边
        df_click = df_click.astype({'user_id': 'str', 'item_id': 'str'}, copy=False)
        index = [i for i,item in enumerate(df_click.item_id)  if item not in self.item_dic.keys()]
        df_click.drop(index, inplace=True)  # 如果样本里没有这个item，则删除
        df_click.reset_index(drop=True, inplace=True)
        print('df_click:', df_click.shape)


        self.df_user=df_user
        self.df_item=df_item
        self.df_click=df_click
        self.df_order=df_order
        self.data_dir=data_dir


    def generate_pure_graph(self,is_save=True):

        import pickle
        u={v:k for k,v in self.user_dic.items()}
        i={v:k for k,v in self.item_dic.items()}
        pickle.dump(u,open('{}/attweight/user_dic.pkl'.format(args.data_dir),'wb'))
        pickle.dump(i,open('{}/attweight/item_dic.pkl'.format(args.data_dir),'wb'))
        exit()
        click_user = [self.user_dic[user] for user in self.df_click.user_id]
        click_item = [self.item_dic[item] for item in self.df_click.item_id]
        purchase_user = [self.user_dic[user] for user in self.df_order.user_id]
        purchase_item = [self.item_dic[item] for item in self.df_order.item_id]
        data_dict = {
            ('user', 'click', 'item'): (torch.tensor(click_user), torch.tensor(click_item)),
            ('item', 'click-by', 'user'): (torch.tensor(click_item), torch.tensor(click_user)),
            ('user', 'purchase', 'item'): (torch.tensor(purchase_user), torch.tensor(purchase_item)),
            ('item', 'purchase-by', 'user'): (torch.tensor(purchase_item), torch.tensor(purchase_user)),
                    }
        G = dgl.heterograph(data_dict)


        model = fasttext.load_model('../data/fasttext/fastText/cc.zh.200.bin')
        temp = {k: model.get_sentence_vector(v) for v, k in self.cid1_dic.items()}
        cid1_feature = torch.tensor([temp[k] for _, k in self.cid1_dic.items()])

        temp = {k: model.get_sentence_vector(v) for v, k in self.cid2_dic.items()}
        cid2_feature = torch.tensor([temp[k] for _, k in self.cid2_dic.items()])

        temp = {k: model.get_sentence_vector(v) for v, k in self.cid3_dic.items()}
        cid3_feature = torch.tensor([temp[k] for _, k in self.cid3_dic.items()])

        temp = {k: model.get_sentence_vector(v) for v, k in self.brand_dic.items()}
        brand_feature = torch.tensor([temp[k] for _, k in self.brand_dic.items()])


        '''将标签传进label'''
        label_gender = self.df_user.gender
        label_age = [self.age_dic[age] for age in self.df_user.age_range]
        G.nodes['user'].data['gender'] = torch.tensor(label_gender[:G.number_of_nodes('user')])
        G.nodes['user'].data['age'] = torch.tensor(label_age[:G.number_of_nodes('user')])
        G.nodes['item'].data['cid1'] = torch.tensor(self.c1[:G.number_of_nodes('item')])
        G.nodes['item'].data['cid2'] = torch.tensor(self.c2[:G.number_of_nodes('item')])
        G.nodes['item'].data['cid3'] = torch.tensor(self.c3[:G.number_of_nodes('item')])
        G.nodes['item'].data['brand'] = torch.tensor(self.brand[:G.number_of_nodes('item')])
        if is_save == True:
            torch.save(G, '{}/G_ori.pkl'.format(self.data_dir))
            torch.save(cid1_feature, '{}/cid1_feature.npy'.format(self.data_dir))
            torch.save(cid2_feature, '{}/cid2_feature.npy'.format(self.data_dir))
            torch.save(cid3_feature, '{}/cid3_feature.npy'.format(self.data_dir))
            torch.save(brand_feature, '{}/brand_feature.npy'.format(self.data_dir))
        self.G = G
        return G


    def generate_graph(self,is_save=True):


        click_cnt=self.df_click['user_id'].value_counts()      # 至少点过10个商品的用户
        click_cnt=click_cnt[click_cnt>=10].to_dict().keys()
        df_click = self.df_click[self.df_click['user_id'].isin(click_cnt)]
        df_click.reset_index(drop=True,inplace=True)
        click_user_item=[(df_click.at[i,'user_id'],df_click.at[i,'item_id']) for i in range(len(df_click))]
        click_user,click_item=list(zip(*click_user_item))


        order_cnt=self.df_order['user_id'].value_counts()
        order_cnt=order_cnt[order_cnt>=10].to_dict().keys()
        df_order = self.df_order[self.df_order['user_id'].isin(order_cnt)]
        df_order.reset_index(drop=True, inplace=True)
        order_user_item=[(df_order.at[i,'user_id'],df_order.at[i,'item_id']) for i in range(len(df_order))]
        purchase_user, purchase_item = list(zip(*order_user_item))

        all_user=set(click_user+purchase_user)
        print('user数量：',len(all_user))
        all_item=set(click_item+purchase_item)
        user_dic={k:v for v,k in enumerate(all_user)}
        df_user=self.df_user[self.df_user['user_id'].isin(all_user)]
        df_user.reset_index(drop=True, inplace=True)
        item_dic={k:v for v,k in enumerate(all_item)}
        df_item=self.df_item[self.df_item['item_id'].isin(all_item)]
        df_item.reset_index(drop=True, inplace=True)


        click_user = [user_dic[user] for user in click_user ]
        click_item = [item_dic[item] for item in click_item]
        purchase_user = [user_dic[user] for user in purchase_user]
        purchase_item = [item_dic[item] for item in purchase_item]



        data_dict = {
            ('user', 'click', 'item'): (torch.tensor(click_user), torch.tensor(click_item)),
            ('item', 'click-by', 'user'): (torch.tensor(click_item), torch.tensor(click_user)),
            ('user', 'purchase', 'item'): (torch.tensor(purchase_user), torch.tensor(purchase_item)),
            ('item', 'purchase-by', 'user'): (torch.tensor(purchase_item), torch.tensor(purchase_user)),

        }
        G = dgl.heterograph(data_dict)

        model = fasttext.load_model('../data/fasttext/fastText/cc.zh.200.bin')
        temp = {k: model.get_sentence_vector(v) for v, k in self.cid1_dic.items()}
        cid1_feature = torch.tensor([temp[k] for _, k in self.cid1_dic.items()])

        temp = {k: model.get_sentence_vector(v) for v, k in self.cid2_dic.items()}
        cid2_feature = torch.tensor([temp[k] for _, k in self.cid2_dic.items()])

        temp = {k: model.get_sentence_vector(v) for v, k in self.cid3_dic.items()}
        cid3_feature = torch.tensor([temp[k] for _, k in self.cid3_dic.items()])

        temp = {k: model.get_sentence_vector(v) for v, k in self.brand_dic.items()}
        brand_feature = torch.tensor([temp[k] for _, k in self.brand_dic.items()])
        '''将标签传进label'''
        # label
        label_gender = df_user.gender
        label_age = [self.age_dic[age] for age in df_user.age_range]
        G.nodes['user'].data['gender'] = torch.tensor(label_gender[:G.number_of_nodes('user')])
        G.nodes['user'].data['age'] = torch.tensor(label_age[:G.number_of_nodes('user')])
        G.nodes['item'].data['cid1'] = torch.tensor(self.c1[:G.number_of_nodes('item')])
        G.nodes['item'].data['cid2'] = torch.tensor(self.c2[:G.number_of_nodes('item')])
        G.nodes['item'].data['cid3'] = torch.tensor(self.c3[:G.number_of_nodes('item')])
        G.nodes['item'].data['brand'] = torch.tensor(self.brand[:G.number_of_nodes('item')])
        if is_save == True:
            torch.save(G, '{}/G.pkl'.format(self.data_dir))
            torch.save(cid1_feature, '{}/cid1_feature.npy'.format(self.data_dir))
            torch.save(cid2_feature, '{}/cid2_feature.npy'.format(self.data_dir))
            torch.save(cid3_feature, '{}/cid3_feature.npy'.format(self.data_dir))
            torch.save(brand_feature, '{}/brand_feature.npy'.format(self.data_dir))

        self.G=G
        return G


    def generate_homo_graph(self,is_save=True):
        # 获得item的属性
        t = time.time()
        item_word_dic = {}
        all_words = []
        del_index=[]
        for v, k in enumerate(self.df_item.item_id):

            words = split_char(self.df_item.at[v,'item_name'])[:10]  # 将字符串切分，取前10个
            att1 = split_char(self.df_item.at[v,'cid1_name'])  # 将字符串切分，取前10个
            att2 = split_char(self.df_item.at[v,'cid2_name'])  # 将字符串切分，取前10个
            att3 = split_char(self.df_item.at[v,'cid3_name'])  # 将字符串切分，取前10个
            t_words = att1 + att2 + att3 + words
            t_words=t_words[:15]

            if len(words)<10:
                del_index.append(v)
            else:

                item_word_dic[k] = words  #     {item1_id:[word1,word2...],...}
                all_words.extend(words)
        print('时间1：', time.time() - t)
        new_df_item = self.df_item.drop(del_index)
        new_df_item.reset_index(drop=True, inplace=True)
        items=new_df_item.item_id.tolist()
        new_item_dic={k:v for v,k in enumerate(items)}
        word_dic = {k: v for v, k in enumerate(set(all_words))}
        item_word_dic = {new_item_dic[k]: list(map(lambda x: word_dic[x], words)) for k, words in
                         item_word_dic.items()}  # {item1_id:[index1,index2...],...}
        item_word=[value for key,value in item_word_dic.items()]

        print('时间2：', time.time() - t)



        init_dic = {self.user_dic[user]: [] for user in self.df_user.user_id}
        user_item_dic = neighbormap(self.df_order,init_dic , self.user_dic,new_item_dic)  # {user1:[item1,item2...],...}
        user_item_dic = neighbormap(self.df_click,user_item_dic , self.user_dic,new_item_dic)
        user_item, del_user = filter_sample(5, user_item_dic)
        print('时间3：', time.time() - t)

        new_df_user=self.df_user.drop(del_user)
        new_df_user.reset_index(drop=True,inplace=True)
        users=new_df_user.user_id.tolist()
        new_user_dic={k:v for v,k in enumerate(users)}
        print('删除的item数：{}，删除的user数：{}'.format(len(del_index),len(del_user)))
        print('用户数：',len(user_item))
        u_i=torch.tensor(user_item)  #取出的user的邻居，应是(N1,10)
        i_w=torch.tensor(item_word)  #取出的user的邻居词属性，应是(N2,10)
        out = i_w[u_i]          # N1*10*10
        print('时间4：', time.time() - t)
        print(out.shape)
        model = fasttext.load_model('../data/fasttext/fastText/cc.zh.200.bin')
        word_feature_dic = {k:model[v] for v,k in word_dic.items()}
        word_feature = torch.tensor([word_feature_dic[k] for _,k in word_dic.items()])
        torch.save(word_feature, '{}/word_feature.npy'.format(self.data_dir))
        exit()


        co_purchase_user = combination(self.df_order,users)
        co_purchase_user[0] = [new_user_dic[user] for user in co_purchase_user[0]]
        co_purchase_user[1] = [new_user_dic[user] for user in co_purchase_user[1]]

        co_click_user = combination(self.df_click,users)
        co_click_user[0] = [new_user_dic[user] for user in co_click_user[0]]
        co_click_user[1] = [new_user_dic[user] for user in co_click_user[1]]

        data_dict={
            ('user', 'co-purchase', 'user'): (torch.tensor(co_purchase_user[0]), torch.tensor(co_purchase_user[1])),
            # ('user', 'co-purchase-by', 'user'): (torch.tensor(co_purchase_user[1]), torch.tensor(co_purchase_user[0])),
            ('user', 'co-click', 'user'): (torch.tensor(co_click_user[0]), torch.tensor(co_click_user[1])),
            # ('user', 'co-click-by', 'user'): (torch.tensor(co_click_user[1]), torch.tensor(co_click_user[0])),
        }
        G = dgl.heterograph(data_dict)
        print('时间5：', time.time() - t)
        # label
        label_gender = new_df_user.gender.tolist()
        label_age = [self.age_dic[age] for age in new_df_user.age_range]
        G.nodes['user'].data['gender'] = torch.tensor(label_gender[:G.number_of_nodes('user')])
        G.nodes['user'].data['age'] = torch.tensor(label_age[:G.number_of_nodes('user')])
        G.nodes['user'].data['all_word'] = out[:G.number_of_nodes('user')]
        G = dgl.to_homogeneous(G,ndata=['gender','age','all_word'])
        print('时间6：', time.time() - t)

        model = fasttext.load_model('../data/fasttext/fastText/cc.zh.200.bin')
        word_feature_dic = {k:model[v] for v,k in word_dic.items()}
        word_feature = torch.tensor([word_feature_dic[k] for _,k in word_dic.items()])
        if is_save:
            torch.save(G,'{}/homo_G_1.pkl'.format(self.data_dir))
            torch.save(word_feature, '{}/word_feature_1.npy'.format(self.data_dir))
        return G




if __name__ == '__main__':
    print('文件路径：',args.data_dir)
    data=dataloader(data_dir=args.data_dir)
    G=data.generate_pure_graph(is_save=True)
    # G=data.generate_graph(is_save=True)
    # G=data.generate_homo_graph(is_save=True)
    print(G)

# items,cid1,cid2,cid3,brand=[],[],[],[],[]
# for i in range(len(df_item)):
#     item=df_item.at[i,'item_id']
#     if item in item_dic:
#         c1 = self.df_item.at[i, 'cid1']
#         c2 = self.df_item.at[i, 'cid2']
#         c3 = self.df_item.at[i, 'cid3']
#         b = self.df_item.at[i, 'brand_code']
#         items.append(item_dic[item])
#         cid1.append(self.cid1_dic[c1])
#         cid2.append(self.cid2_dic[c2])
#         cid3.append(self.cid3_dic[c3])
#         brand.append(self.brand_dic[b])

# def show():
    # # 展示比例
    # count_dic={'flag':0}
    # for key in neighbor_dic:
    #     length=len(neighbor_dic[key])
    #     if length>=10:
    #         count_dic['flag']+=1
    #     if length not in count_dic:
    #         count_dic[length]=1
    #     else:
    #         count_dic[length]+=1
    # for key in count_dic:
    #     percent = (count_dic[key]/len(neighbor_dic.keys()))*100
    #     if percent > 0.1:
    #         print('邻居数：{}  占比：{} % 数量为：{}'.format(key,round(percent,2),count_dic[key]))
    # print('节点数共计:',len(neighbor_dic))
    # print('运行时间为：',time.time()-t,'秒')
    # pass
