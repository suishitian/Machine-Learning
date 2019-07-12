from math import log
import operator
import copy

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

class Tree:

    def __init__(self,dataSet,labels):
        self.dataSet = dataSet
        self.labels = labels
        self.myTree = {}
        self.builded = False

    @staticmethod
    def calShannonEnt(dataSet):
        #计算一个数据集的香农熵
        #公式: H=-求和(p(xi)log2(p(xi))) 其中p(xi)为xi标签的概率，实际上为xi的数量/总数量即可
        #在对所有标签的熵值求和(负号)，为最后的结果
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            #默认数据每一行的最后一个值是标签值
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            #求每一个key标签的概率
            prob = float(labelCounts[key])/numEntries
            #求log值，并且求和，因为前面是负号，所以是-=
            shannonEnt -= prob*log(prob,2)
        return shannonEnt

    def splitDataSet(self,dataSet,feature_index,value):
        #该函数的目的是将所有样本中第feature_index个特征的值等于value的样本抽取出来
        retDataSet = list()
        for vec in dataSet:
            if vec[feature_index] == value:
                #抽取出来的样本应该不包含feature_index指向的特征，所以在样本向量中将该特征删除(缩减)
                reducedFeatVec = vec[:feature_index]
                reducedFeatVec.extend(vec[feature_index+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self,dataSet):
        numFeature = len(dataSet[0]) - 1
        baseEntropy = Tree.calShannonEnt(dataSet)
        bestInfoGain = 0.0
        bestFeatureIndex = -1
        #对每一种特征进行信息增益计算
        for i in range(numFeature):
            #featureValueVec为每一个样本的对应特征的值组成的列表(可以理解为某特征对应的列的列表)
            featureValueVec = [example[i] for example in dataSet]
            #将某特征取值的列表，进行set操作，得到该特征的所有可能取值，用作样本抽取
            featureValueSet = set(featureValueVec)
            #遍历每一个特征取值，对每一个取值进行抽取，并且计算熵
            new_entropy = 0.0
            for value in featureValueSet:
                extractedVec = self.splitDataSet(dataSet,i,value)
                #计算抽取的数据的频率
                prob = len(extractedVec)/float(len(dataSet))
                #频率*子集的熵，然后求和，此值为切分完特征之后的最终熵值
                new_entropy += prob * Tree.calShannonEnt(extractedVec)
            #将某特征的最终熵值与原熵进行比较，最小的熵(与原熵的差最大)，表示混乱程度最低。则是最优特征
            infoGain = baseEntropy - new_entropy
            if infoGain>bestInfoGain:
                bestInfoGain = infoGain
                bestFeatureIndex = i
        return bestFeatureIndex

    def majorityCnt(self,classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        #classCount.items()返回的是二维元组
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

    def createTree(self,dataSet,labels):
        #创建数据集的标签(最后一项)的所有值列表
        classList = [example[-1] for example in dataSet]
        #列表中第一项的数量等于列表的size，即每一项都等于第一项，即类别是统一的
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        #只有一个值，说明所有特征都被分配完了，但是还是没有所有样本都属于同一类
        if len(dataSet[0]) == 1:
            #这种情况下使用投票法，作为这一个子集的类别标签
            return self.majorityCnt(classList)
        #选出最佳切分特征
        bestFeatureIndex = self.chooseBestFeatureToSplit(dataSet)
        #最佳切分特征对应的lebals值
        bestFeatureLabel = labels[bestFeatureIndex]
        #当前层创建树(会返回给上一层)
        myTree = {bestFeatureLabel:{}}
        #在labels中删除选中的特征(特征删减要与dataSet保持一致)
        del(labels[bestFeatureIndex])
        #生成选中特征的取值列表
        featureValue = [example[bestFeatureIndex] for example in dataSet]
        #进行set操作
        featureSet = set(featureValue)
        #遍历选中特征的每一个取值
        for value in featureSet:
            #根据每一个取值抽取对应的数据子集
            extractedData = self.splitDataSet(dataSet,bestFeatureIndex,value)
            #缩减之后列表的拷贝(因为后续有很多子层会直接使用传入的labels)
            new_labels = labels[:]
            #在当前层中bestFeatureLable特征的每一个取值对应的值(树)为使用子数据集和子标签的createTree构建树
            myTree[bestFeatureLabel][value] = self.createTree(extractedData,new_labels)
        #返回当前层的树给上一层(如果为根节点，则myTree为整个树)
        return myTree

    def BuildTree(self):
        #在保持类内数据不变的情况下，进行深拷贝，然后在进行构建
        dataSet = copy.deepcopy(self.dataSet)
        labels = copy.deepcopy(self.labels)
        self.myTree = self.createTree(dataSet,labels)
        self.builded = True


if __name__=="__main__":
    dataSet,labels = createDataSet()
    #dataSet[0][-1] = 'hahah'
    print(dataSet)
    shannonEnt = Tree.calShannonEnt(dataSet)
    print(shannonEnt)
    tree = Tree(dataSet,labels)
    mat = tree.splitDataSet(tree.dataSet,2,'yes')
    print(mat)
    res = tree.chooseBestFeatureToSplit(tree.dataSet)
    print(res)
    tree.BuildTree()
    print(tree.myTree)
