from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLine = len(arrayLines)
    returnMat = zeros((numberOfLine,3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromline = line.split('\t')
        #前三个为特征，后一个为labels
        returnMat[index,:] = listFromline[0:3]
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat, classLabelVector

def generatePlot(data,labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:,0],data[:,1],15.0*array(labels),15.0*array(labels))
    plt.show()

class KNN:
    def __init__(self, k_value, dataSet, labels):
        self.k_value = k_value
        self.dataSet = dataSet
        self.labels = labels
        self.loaded = True

    def autoNorm(self):
        #归一化公式：newValue = (oldValue-min)/(max-min) 其中min,max为当前列的最大最小
        if self.loaded == False:
            return False,[]
        #.min(0)表示是对列求最小值，返回的是(1*列数)的数据结构(每一列的最小值)
        minVals = self.dataSet.min(0)
        maxVals = self.dataSet.max(0)
        ranges = maxVals - minVals
        #获取行数
        m = self.dataSet.shape[0]
        #将每一列的最小值，在每一行复制，扩展为与原数据结构一样的矩阵，方便每一项做相减运算
        normDataSet = self.dataSet - tile(minVals,(m,1))
        normDataSet = normDataSet/tile(ranges,(m,1))  #这里的除法不是矩阵除法
        self.dataSet = normDataSet
        print(self.dataSet)

    def process(self,input):
        if self.loaded == False:
            return False,0
        dataSetSize = self.dataSet.shape[0]
        labels = self.labels
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
        return True,sortedClassCount[0][0]

#group, labels = createDataSet()
group, labels = file2matrix("./data/datingTestSet2.txt")
#knn = KNN(2, group, labels)
#flag, result = knn.process([0,0.2])
print(group)
print(labels[:20])
#generatePlot(group,labels)
knn = KNN(2,group,labels)
knn.autoNorm()
generatePlot(knn.dataSet,knn.labels)