#!/usr/bin/python
#coding:utf-8

from collections import defaultdict
from itertools import chain


class CharDict(object):

    def __init__(self, words, split='#'):
        self._char_dict = defaultdict(int)
        char_set = set(chain(*(w.strip() for w in words)))
        if split in char_set:
            raise Exception('split character is ilegal.')
        self._split_character = split

        start = 1
        self._char_dict['#'] = start
        for index, item in enumerate(char_set):
            self._char_dict[item] = index + start + 1

    @property
    def split(self):
        return self._split_character

    def __getitem__(self, key):
        return self._char_dict[key]

    def __len__(self):
        return len(self._char_dict)

    def add_word(self, word):
        for c in word:
            if not self._char_dict[c]:
                self._char_dict[c] = len(self._char_dict) + 1

    def iteritems(self):
        return self._char_dict.items()

    def iterkeys(self):
        return self._char_dict.keys()

    def __str__(self):
        return str(self._char_dict)


class DoubleArrayTrie(object):

    def __init__(self, words, split='#'):
        self.chars = CharDict(words, split)
        self.base = defaultdict(int)
        self.check = defaultdict(int)
        self.tails = defaultdict(int)
        self.next_pos = 1  # tails中下一个可插入的位置
        self.root = 1   # root节点
        for word in words:
            self.insert(word)

    def __str__(self):
        return 'chars: {}\nbase: {}\ncheck: {}\ntails: {}'.format(str(self.chars), str(self.base), str(self.check), str(self.tails))

    def check_word(self, word):
        return False if '#' in word else True

    def compare_tails(self, start, word, i):
        pos = -self.base[start]
        tail1 = self.tails[pos]
        tail2 = word[i+1:] + '#'  # 通过'#'字符来规避一个字符串是另一个字符串的子串的情况
        compare_result = True if tail1 == tail2 else False
        return compare_result, pos, tail1, tail2

    def search(self, word):
        if not self.check_word(word):
            return 'word ilegal', None
        start = self.root
        for i, c in enumerate(word):
            arc = self.chars[c]
            end = self.base[start] + arc
            if self.check[end] == 0:
                return 'not exists', None
            elif self.check[end] != start:
                return 'exists part prefix', self.get_node_strs(word[:i+1], start)
            else:
                start = end
                if self.base[start] < 0:
                    compare_result, pos, tail1, _ = self.compare_tails(start, word, i)
                    if compare_result:
                        return 'exists', [word,]
                    return 'exists part prefix', [word[:i+1] + tail1[:-1],]
        end = self.base[start] + self.chars['#']
        if self.check[end] != start:
            return 'exists prefix', self.get_node_strs(word, start)
        else:
            return 'exists', [word,]

    def get_node_strs(self, prefix, start):
        '''获取所有经过start节点的字符串，prefix是start节点之前的边组成的前缀字符串'''
        result = []
        for tail in self.get_node_tails(start):
            result.append(prefix+tail[:-1])
        return result

    def get_node_tails(self, start):
        '''获取所有以start节点为开始的后缀字符串'''
        if self.base[start] < 0:
            return [self.tails[-self.base[start]],]
        else:
            arcs, c_list = self.find_arcs(start)
            result = []
            for arc,c in zip(arcs, c_list):
                end = self.base[start] + arc
                tails = self.get_node_tails(end)
                for tail in tails:
                    result.append(c+tail)
            return result

    def delete(self, word):
        if not self.check_word(word):
            return False
        start = self.root
        for i, c in enumerate(word):
            arc = self.chars[c]
            end = self.base[start] + arc
            if self.check[end] == 0 or self.check[end] != start:
                return False
            else:
                start = end
                if self.base[start] < 0:
                    compare_result, pos, _, _ = self.compare_tails(start, word, i)
                    if compare_result:
                        self.tails.pop(pos)
                        self.base[start] = self.check[start] = 0
                        return True
                    return False
        end = self.base[start] + self.chars['#']
        if self.check[end] == start:
            self.tails.pop(-self.base[end])
            self.base[end] = self.check[end] = 0
            return True
        else:
            return False


    def insert(self, word):
        if not self.check_word(word):
            return False
        self.chars.add_word(word)
        start = self.root
        for i, c in enumerate(word):
            arc = self.chars[c]
            end = self.base[start] + arc
            if self.check[end] == 0:
                # base[end]将会变成独立节点，后续字符串将会保存到tails中
                self.write_tail(start, end, word[i+1:]+'#')
                return True
            elif self.check[end] == start:
                # 从start到end的有向边已存在
                start = end
                if self.base[start] < 0:
                    # start是独立节点，需要比较word的后续字符串和tails中的字符串
                    compare_result, pos, tail1, tail2 = self.compare_tails(start, word, i)
                    if compare_result:
                        # 该word已存在
                        return True

                    lcp = self.longest_common_prefix(tail1, tail2)  # 找到最长公共子串
                    if lcp is not '':
                        q = self.x_check(lcp)
                        for new_c in lcp:  # 存储公共前缀字符串
                            self.base[start] = q
                            # 下一个节点
                            start = self.add_arc(new_c, start)

                    len_lcp = len(lcp)
                    # x != y 一定成立
                    x, y = tail1[len_lcp:], tail2[len_lcp:]
                    q = self.x_check(x[0]+y[0])

                    self.base[start] = q
                    # 将x保存到tails中
                    end = self.add_arc(x[0], start)
                    self.base[end] = -pos
                    self.tails[pos] = x[1:]
                    # 将y保存到tails中
                    end = self.add_arc(y[0], start)
                    self.write_tail(start, end, y[1:])  # 因为返回的tail2中已包含'#'所以此处不需要再加
                    return True
            else:
                # 从start到end的有向边不存在，此时发生冲突
                old_start = self.check[end]
                conflict = {
                    start: self.find_arcs(start),
                    old_start: self.find_arcs(old_start),
                }
                if len(conflict[start][0]) + 1 > len(conflict[old_start][0]):
                    # 更改冲突较小的节点
                    change_node = old_start
                else:
                    change_node = start
                temp_base = self.base[change_node]
                q = self.x_check(conflict[change_node][1])
                self.base[change_node] = q
                for arc in conflict[change_node][0]:
                    # 转移冲突的节点
                    old_end = temp_base + arc
                    new_end = self.base[change_node] + arc
                    self.base[new_end] = self.base[old_end]
                    self.check[new_end] = self.check[old_end]
                    if self.base[old_end] > 0:
                        # 当old_end是其他节点的父节点时，需要将它对应的子节点对应的父节点更改为new_end
                        for key, value in self.check.items():  # 是否需要如同find_arcs方法中，使用遍历chars的方法？
                            if value == old_end:
                                self.check[key] = new_end
                    self.base[old_end] = self.check[old_end] = 0

                self.write_tail(start, end, word[i+1:]+'#')
                return True

    def longest_common_prefix(self, *words):
        '''返回多个字符串的最长公共子串'''
        lcp = ''
        for item in zip(*words):
            if len(set(item)) == 1:
                lcp += item[0]
            else:
                break
        return lcp

    def x_check(self, word):
        '''
        在base数组中发生冲突时，找到最近一个基准位置，确保有足够的空位保存冲突节点.
        即，要找到一个赋值给base[start]的值，能够保证所有对应的冲突节点对应的check[end]都为0.
        '''
        q = 1
        while 1:
            if all(self.check[q + self.chars[c]] == 0 for c in word):
                break
            q += 1
        return q

    def write_tail(self, start, end, word):
        self.base[end] = -self.next_pos
        self.check[end] = start
        self.tails[self.next_pos] = word
        self.next_pos += len(self.tails[self.next_pos])

    def add_arc(self, c, start):
        '''加一条从start节点开始，边对应的字符是c的有向边'''
        arc = self.chars[c]
        end = self.base[start] + arc
        self.check[end] = start
        return end

    def find_arcs(self, start):
        '''找到start节点对应的所有边和所有end节点'''
        arcs, c_list = [], []
#        reverse_chars = {v:k for k,v in self.chars.iteritems()}
#        for key, value in self.check.iteritems():
#            if value == start:
#                arc = key - self.base[start]
#                arcs.append(arc)
#        return arcs
        for c in self.chars.iterkeys():
            arc = self.chars[c]
            if self.check[self.base[start] + arc] == start:
                arcs.append(arc)
                c_list.append(c)
        return arcs, c_list

    def search_exists(self, word):
        pass


if __name__ == '__main__':
    pass
    #
    # words = ['123','234','321']
    # trie = DoubleArrayTrie(words=words)
    #
    # content = "123123167896789234321"
    # temp_content = ""
    # for i in content:
    #     temp_content += i
    #     flag, res = trie.search(temp_content)
    #     print(flag)
    #     if flag == "exists":
    #         print(res)
    #         temp_content = ""
