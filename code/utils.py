import gzip
import hickle
import _pickle as cPickle
import itertools
import time

def get_num_neighbor(G,etype):
    print(G.edges(etype=etype))
    for i in G.edges(etype=etype):
        print(i)
    #     exit()


def neighbormap(df,dic,user_dic,new_item_dic,col_user='user_id',col_item='item_id'):
    t=time.time()
    print('开始了')
    for i in range(len(df)):
        user=df.at[i,col_user]
        item=df.at[i,col_item]
        if item in new_item_dic:
            dic[user_dic[user]].append(new_item_dic[item])

    print('结束时间',time.time()-t)
    return dic

def split_char(str):
    english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    output = []
    buffer = ''
    try:
        for s in str:
            if s in english or s in english.upper(): #英文或数字
                buffer += s
            elif s in ' （）*()【】/-.':         #如果是空格等特殊符号就跳过
                continue
            else: #中文
                if buffer:
                    output.append(buffer)
                buffer = ''
                output.append(s)
        if buffer:
            output.append(buffer)
    except:
        print(str)
    return output



def filter_sample(threshold,dic):

    del_index = []
    out = []
    for key,value in dic.items():
        if len(set(value)) < threshold:
            del_index.append(key)
        else:
            neirghbor = value
            out.append(neirghbor[:threshold])
    return out,del_index

def combination(df,users,col_user='user_id',col_item='item_id'):


    df = df[df[col_user].isin(users)]           #筛选，用户必须是满足条件的用户
    df.reset_index(drop=True, inplace=True)
    df_item=df[col_item].value_counts()
    items = df_item[df_item >= 10].to_dict().keys()  #筛选，item的被点击用户数要大于某个值
    df = df[df[col_item].isin(items)]
    df.reset_index(drop=True, inplace=True)
    print(df.shape,len(list(df.groupby([col_item]))))
    out = []
    for iter in df.groupby([col_item]):
        l = iter[1][col_user].tolist()
        l = [x for x in l if x in set(users)]
        pairs = list(itertools.combinations(l, 2))[:10 if 10>len(l) else len(l)]
        out.extend(pairs)

    out = list(zip(*set(out)))
    print('去重后的边数:', len(out[0]))
    return out
