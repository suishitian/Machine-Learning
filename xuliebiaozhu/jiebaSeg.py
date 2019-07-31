#coding=utf-8
import jieba

#query = "在北京举行的庆祝新中国成立70周年湖南专场新闻发布会"

query = "景区非常好，酒店设施非常新，温泉池非常舒服"

#query = "希望香港社会反对暴力、守护法治"
seg_list = jieba.cut(query, cut_all=True)
print("all: " + "/ ".join(seg_list))


seg_list = jieba.cut(query, cut_all=False, HMM=False)
print("sec: " + "/ ".join(seg_list))
 
seg_list = jieba.cut(query, cut_all=False, HMM=True)
print("HMM: " + "/ ".join(seg_list))


seg_list = jieba.cut_for_search(query, HMM=True)
print("search: " + "/ ".join(seg_list))

query = "设置股票预警"
seg_list = jieba.cut(query, cut_all=True)
print("all: " + "/ ".join(seg_list))
print(jieba.get_DAG(query))
