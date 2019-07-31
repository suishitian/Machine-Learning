import jieba
import re
import DoubleArrayTrie

def maxMatch(input,trie):
    max_length = -1
    max_word = ""
    temp_word = ""
    for i in input:
        temp_word += i
        flag, res = trie.search(temp_word)
        if flag == "exists":
            #print(res)
            if len(res[0]) > max_length:
                max_word = res[0]
                max_length = len(res[0])
    #print("max_word:"+max_word)
    return max_word

def build():
    fW = open('res2.txt', 'r', encoding='UTF-8')
    words = fW.read().strip().split(' ')
    words = list(set(words))
    trie = DoubleArrayTrie.DoubleArrayTrie(words=words)
    return trie

def process(trie,content):
    #content = "我在水哈哈果同花顺苹果"
    unknown_word = ""
    unknown_word_list = list()
    i = 0
    res_list = list()
    while i<len(content):
        temp_content = content[i:]
        maxWord = maxMatch(temp_content,trie)
        if maxWord != "":
            i += len(maxWord)
            if unknown_word != "":
                unknown_word_list.append(unknown_word)
                res_list.append("("+unknown_word+")")
                unknown_word = ""
            res_list.append(maxWord)
        else:
            unknown_word += content[i]
            i += 1
    return unknown_word_list,res_list

if __name__ == "__main__":
    trie = build()
    _, res_list = process(trie, "设置股票预警")
    print(' '.join(res_list))
    _, res_list = process(trie, "股票怎么样")
    print(' '.join(res_list))
    _, res_list = process(trie, "在北京举行的庆祝新中国成立70周年湖南专场新闻发布会")
    print(' '.join(res_list))
    _, res_list = process(trie, "希望香港社会反对暴力、守护法治")
    print(' '.join(res_list))
    _, res_list = process(trie, "景区非常好，酒店设施非常新，温泉池非常舒服")
    print(' '.join(res_list))

