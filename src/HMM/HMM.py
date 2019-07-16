# -*- coding: utf-8 -*-
##摘自网络资源！！
"""
Created on Thu Feb 16 19:28:39 2017
2017-4-2
    ForwardBackwardAlg函数功能：实现前向算法
    理论依据：李航《统计学习方法》
2017-4-5
    修改了ForwardBackwardAlg函数名称为ForwardAlgo以及输出的alpha数组形式
    完成了BackwardAlgo函数功能：后向算法
    以及函数FBAlgoAppli：计算在观测序列和模型参数确定的情况下，
    某一个隐含状态对应相应的观测状态的概率
2017-4-6
    完成BaumWelchAlgo函数一次迭代
2017-4-7
    实现维特比算法
@author: sgp
"""

import numpy as np
import copy

# 输入格式如下：
# A = np.array([[.5,.2,.3],[.3,.5,.2],[.2,.3,.5]])
# B = np.array([[.5,.5],[.4,.6],[.7,.3]])
# Pi = np.array([[.2,.4,.4]])
# O = np.array([[1,2,1]])

# 应用ndarray在数组之间进行相互运算时，一定要确保数组维数相同！
# 比如：
# In[93]:m = np.array([1,2,3,4])
# In[94]:m
# Out[94]: array([1, 2, 3, 4])
# In[95]:m.shape
# Out[95]: (4,)
# 这里表示的是一维数组
# In[96]:m = np.array([[1,2,3,4]])
# In[97]:m
# Out[97]: array([[1, 2, 3, 4]])
# In[98]:m.shape
# Out[98]: (1, 4)
# 而这里表示的就是二维数组
# 注意In[93]和In[96]的区别,多一对中括号！！

# N = A.shape[0]为数组A的行数， H = O.shape[1]为数组O的列数
# 在下列各函数中，alpha数组和beta数组均为N*H二维数组，也就是横向坐标是时间，纵向是状态

def ForwardAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    H = O.shape[1]  # 数组O的列数

    sum_alpha_1 = np.zeros((M, N))
    alpha = np.zeros((N, H))
    r = np.zeros((1, N))
    alpha_1 = np.multiply(Pi[0, :], B[:, O[0, 0] - 1])

    # alpha_1是一维数组，在使用np.multiply的时候需要升级到二维数组。#错误是IndexError: too many indices for array
    alpha[:, 0] = np.array(alpha_1).reshape(1,N)

    for h in range(1, H):
        for i in range(N):
            for j in range(M):
                sum_alpha_1[i, j] = alpha[j, h - 1] * A[j, i]
            r = sum_alpha_1.sum(1).reshape(1, N)  # 同理，将数组升级为二维数组
            alpha[i, h] = r[0, i] * B[i, O[0, h] - 1]
    # print("alpha矩阵: \n %r" % alpha)
    p = alpha.sum(0).reshape(1, H)
    P = p[0, H - 1]
    # print("观测概率: \n %r" % P)
    # return alpha
    return alpha, P


def BackwardAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    H = O.shape[1]  # 数组O的列数

    # beta = np.zeros((N,H))
    sum_beta = np.zeros((1, N))
    beta = np.zeros((N, H))
    beta[:, H - 1] = 1
    p_beta = np.zeros((1, N))

    for h in range(H - 1, 0, -1):
        for i in range(N):
            for j in range(M):
                sum_beta[0, j] = A[i, j] * B[j, O[0, h] - 1] * beta[j, h]
            beta[i, h - 1] = sum_beta.sum(1)
    # print("beta矩阵: \n %r" % beta)
    for i in range(N):
        p_beta[0, i] = Pi[0, i] * B[i, O[0, 0] - 1] * beta[i, 0]
    p = p_beta.sum(1).reshape(1, 1)
    # print("观测概率: \n %r" % p[0,0])
    return beta, p[0, 0]


def FBAlgoAppli(A, B, Pi, O, I):
    # 计算在观测序列和模型参数确定的情况下，某一个隐含状态对应相应的观测状态的概率
    # 例题参考李航《统计学习方法》P189习题10.2
    # 输入格式：
    # I为二维数组，存放所求概率P(it = qi,O|lambda)中it和qi的角标t和i，即P=[t,i]
    alpha, p1 = ForwardAlgo(A, B, Pi, O)
    beta, p2 = BackwardAlgo(A, B, Pi, O)
    p = alpha[I[0, 1] - 1, I[0, 0] - 1] * beta[I[0, 1] - 1, I[0, 0] - 1] / p1
    return p


def GetGamma(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    H = O.shape[1]  # 数组O的列数
    Gamma = np.zeros((N, H))
    alpha, p1 = ForwardAlgo(A, B, Pi, O)
    beta, p2 = BackwardAlgo(A, B, Pi, O)
    for h in range(H):
        for i in range(N):
            Gamma[i, h] = alpha[i, h] * beta[i, h] / p1
    return Gamma


def GetXi(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    H = O.shape[1]  # 数组O的列数
    Xi = np.zeros((H - 1, N, M))
    alpha, p1 = ForwardAlgo(A, B, Pi, O)
    beta, p2 = BackwardAlgo(A, B, Pi, O)
    for h in range(H - 1):
        for i in range(N):
            for j in range(M):
                Xi[h, i, j] = alpha[i, h] * A[i, j] * B[j, O[0, h + 1] - 1] * beta[j, h + 1] / p1
    # print("Xi矩阵: \n %r" % Xi)
    return Xi


def BaumWelchAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    Y = B.shape[1]  # 数组B的列数
    H = O.shape[1]  # 数组O的列数
    c = 0
    Gamma = GetGamma(A, B, Pi, O)
    Xi = GetXi(A, B, Pi, O)
    Xi_1 = Xi.sum(0)
    a = np.zeros((N, M))
    b = np.zeros((M, Y))
    pi = np.zeros((1, N))
    a_1 = np.subtract(Gamma.sum(1), Gamma[:, H - 1]).reshape(1, N)
    for i in range(N):
        for j in range(M):
            a[i, j] = Xi_1[i, j] / a_1[0, i]
    # print(a)
    for y in range(Y):
        for j in range(M):
            for h in range(H):
                if O[0, h] - 1 == y:
                    c = c + Gamma[j, h]
            gamma = Gamma.sum(1).reshape(1, N)
            b[j, y] = c / gamma[0, j]
            c = 0
    # print(b)
    for i in range(N):
        pi[0, i] = Gamma[i, 0]
    # print(pi)
    return a, b, pi


def BaumWelchAlgo_n(A, B, Pi, O, n):  # 计算迭代次数为n的BaumWelch算法
    A_current = copy.deepcopy(A)
    B_current = copy.deepcopy(B)
    Pi_current = copy.deepcopy(Pi)
    O_current = copy.deepcopy(O)
    for i in range(n):
        A_current, B_current, Pi_current = BaumWelchAlgo(A_current, B_current, Pi_current, O_current)
    return A_current, B_current, Pi_current


def viterbi(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行数
    M = A.shape[1]  # 数组A的列数
    H = O.shape[1]  # 数组O的列数
    Delta = np.zeros((M, H))
    Psi = np.zeros((M, H))
    Delta_1 = np.zeros((N, 1))
    I = np.zeros((1, H))

    for i in range(N):
        Delta[i, 0] = Pi[0, i] * B[i, O[0, 0] - 1]

    for h in range(1, H):
        for j in range(M):
            for i in range(N):
                Delta_1[i, 0] = Delta[i, h - 1] * A[i, j] * B[j, O[0, h] - 1]
            Delta[j, h] = np.amax(Delta_1)
            Psi[j, h] = np.argmax(Delta_1) + 1
    print("Delta矩阵: \n %r" % Delta)
    print("Psi矩阵: \n %r" % Psi)
    P_best = np.amax(Delta[:, H - 1])
    psi = np.argmax(Delta[:, H - 1])
    I[0, H - 1] = psi + 1
    for h in range(H - 1, 0, -1):
        I[0, int(h - 1)] = Psi[int(I[0, h] - 1), int(h)]
    print("最优路径概率: \n %r" % P_best)
    print("最优路径: \n %r" % I)

if __name__=="__main__":
    A = np.array([[.5,.2,.3],
                  [.3,.5,.2],
                  [.2,.3,.5]])
    B = np.array([[.5,.5],
                  [.4,.6],
                  [.7,.3]])
    Pi = np.array([[.2,.4,.4]])
    O = np.array([[1,2,1]])
    print("前向后向概率计算")
    alpha, P = ForwardAlgo(A, B, Pi, O)
    print("前向概率矩阵:")
    print(alpha)
    print("前向概率计算的观测概率值:")
    print(P)
    alpha_b, P_b = BackwardAlgo(A, B, Pi, O)
    print("后向概率矩阵:")
    print(alpha_b)
    print("后向概率计算的观测概率值:")
    print(P_b)
    A_s = np.array([[.5, .5, .5],
                  [.5, .5, .5],
                  [.5, .5, .5]])
    B_s = np.array([[.5, .5],
                  [.5, .5],
                  [.5, .5]])
    Pi_s = np.array([[.5, .5, .5]])
    O = np.array([[1, 2, 1]])
    print()
    print("viterbi算法:")
    viterbi(A, B, Pi, O)
    A = np.array([[.5, .2, .3],
                  [.3, .5, .2],
                  [.2, .3, .5]])
    B = np.array([[.5, .5],
                  [.4, .6],
                  [.7, .3]])
    Pi = np.array([[.2, .4, .4]])
    O = np.array([[1, 2, 1]])
    print()
    print("BaumWelch算法:")
    A_new,B_new,C_new = BaumWelchAlgo_n(A, B, Pi, O,5)
    print("A:")
    print(A_new)
    print("C:")
    print(B_new)
    print("Pi:")
    print(C_new)
