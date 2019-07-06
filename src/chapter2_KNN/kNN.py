from numpy import *
import operator
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

class KNN:
    def __init__(self, k_value, dataSet, labels):
        self.k_value = k_value
        self.dataSet = dataSet
        self.labels = labels

    def loadData(self,filename):
        fr = open(filename)
        arrayLines = fr.readlines()
        numberOfLine = len(arrayLines)
        returnMat = zeros((numberOfLine,3))
        classLabelVector = []
        index = 0
        for line in arrayLines:
            line = line.strip()
            listFromline = line.split('\t')
            returnMat[index,:] = listFromline[0:3]
            classLabelVector.append(int(listFromline[-1]))
            index += 1
        self.dataSet = returnMat
        self.labels = classLabelVector

    def process(self,input):
        dataSetSize = self.dataSet.shape[0]
        ##计算输入与所有其他坐标点的距离
        diffMat = tile(input, (dataSetSize,1)) - self.dataSet  #将输入整理为与训练数据一样维度，然后就相减
        sqDiffMat = diffMat**2  #对应维度距离做差之后平方
        sqDistances = sqDiffMat.sum(axis=1)  #平方和
        distances = sqDistances**0.5  #再开方，完成距离公式
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i in range(self.k_value):
            voteIlabel = labels[sortedDistIndicies[i]]
            # 统计前k个距离最近的样本的对应类别
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        return sortedClassCount[0][0]

group, labels = createDataSet()
knn = KNN(2, group, labels)
result = knn.process([0,0.2])
print(result)