import numpy as np
def loadDataSet():
    postingList = [
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','I','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    classVes = [0,1,0,1,0,1]
    return postingList,classVes
class NaiveBayes:
    def __init__(self,dataSet=None, labels=None):
        self.dataSet = dataSet
        self.labels = labels
        self.vocab = None

    @staticmethod
    def createVocabList(dataSet):
        #创建一个空set
        vocabSet = set([])
        for record in dataSet:
            #'|'用于求两个集合的并集
            vocabSet = vocabSet | set(record)
        return list(vocabSet)

    @staticmethod
    def setOfWords2Vec(vocabList, inputSet):
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else:
                print("the word: %s is not in my vocab"%word)
        return returnVec

    @staticmethod
    def makeTrainVec(trainMat,labels):
        numTrainDocs = len(trainMat)
        numWords = len(trainMat[0])
        #计算类别为1的文档的概率(用标签为1的数量除以总数量)
        p_class1 = sum(labels)/float(numTrainDocs)
        c0_num = np.zeros(numWords)
        c1_num = np.zeros(numWords)
        c0_total = 0.0
        c1_total = 0.0
        for i in range(numTrainDocs):
            if labels[i] == 1:
                c1_num += trainMat[i]
                c1_total += sum(trainMat[i])
            else:
                c0_num += trainMat[i]
                c0_total += sum(trainMat[i])
        c1_vec = c1_num/c1_total
        c0_vec = c0_num/c0_total
        return c0_vec,c1_vec,p_class1

    def buildVocab(self):
        if len(self.dataSet)==None:
            print("there is no date in class")
            return
        self.vocab = NaiveBayes.createVocabList(self.dataSet)

    def makeWordVec(self,input):
        if self.vocab==None:
            print("there is no date in class")
            return list()
        return NaiveBayes.setOfWords2Vec(self.vocab,input)

    def buildTrainMat(self):
        if self.dataSet == None:
            

if __name__=='__main__':
    a,b = loadDataSet()
    vocab = NaiveBayes.createVocabList(a)
    print(vocab)
    chain = NaiveBayes.setOfWords2Vec(vocab, a[1])
    print(chain)