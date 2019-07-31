import numpy as np
from math import *
import random

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

def textParse(text):
    import re
    listOfTokens = re.split(r'\W*',text)
    return [token.lower() for token in listOfTokens if len(token)>2]

class NaiveBayes:
    def __init__(self,dataSet=None, labels=None):
        self.dataSet = dataSet
        self.labels = labels
        self.vocab = None
        self.trainMat = None
        self.vec1 = None
        self.vec2 = None
        self.testMat = None
        self.testLabels = None

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
    def bagOfWords2Vec(vocabList,inputSet):
        returnVec = [0] * len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] += 1
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

    def buildTrainMat(self,is_random=False):
        if self.dataSet == None or self.vocab == None:
            print("no dataset and vocab")
            return
        self.trainMat = list()
        if is_random:
            indexList = range(len(self.dataSet)-1)
            randIndex = random.sample(indexList, int(len(self.dataSet) * 0.1))
            self.testMat = list()
            self.testLabels = list()
            for i in randIndex:
                print(i)
                self.testMat.append(self.dataSet[i])
                self.testLabels.append(self.labels[i])
            newDataSet = list()
            newLabel = list()
            for i in range(len(self.dataSet)):
                if i not in randIndex:
                    newDataSet.append(self.dataSet[i])
                    newLabel.append(self.labels[i])
            self.dataSet = newDataSet
            self.labels = newLabel
        for vec in self.dataSet:
            self.trainMat.append(NaiveBayes.setOfWords2Vec(self.vocab,vec))
        v1,v2,p1 = NaiveBayes.makeTrainVec(self.trainMat,self.labels)
        self.vec1 = v1
        self.vec2 = v2
        self.pClass1 = p1

    def train(self,is_random=False):
        #if self.dataSet == None or self.labels == None or self.vocab == None:
        #    print("No data")
        #    return
        self.buildVocab()
        self.buildTrainMat(is_random=is_random)

    def test(self):
        if self.testLabels == None or self.testMat==None:
            print("no test data")
            return
        errorcount = 0
        for i in range(len(self.testMat)):
            res = self.process(self.testMat[i])
            if res!=self.testLabels[i]:
                errorcount+=1
        print("error rate is %f"%(errorcount/len(self.testMat)))

    def process(self,input,is_transfer=True):
        if is_transfer:
            input = self.makeWordVec(input)
        p0 = sum(input*self.vec1)+log(self.pClass1)
        p1 = sum(input*self.vec2)+log(1.0-self.pClass1)
        if p1>p0:
            print("1")
            return 1
        else:
            print("0")
            return 0


def generateDataSet():
    docList = list()
    classList = list()
    fullText = list()
    for i in range(1,26):
        #print("./data/email/spam/%d.txt"%i)
        wordList = textParse(open('./data/email/spam/%d.txt'%i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        #print("./data/email/ham/%d.txt" % i)
        wordList = textParse(open('./data/email/ham/%d.txt'%i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    return docList,classList




if __name__=='__main__':
    a,b = loadDataSet()
    vocab = NaiveBayes.createVocabList(a)
    print(vocab)
    chain = NaiveBayes.setOfWords2Vec(vocab, a[1])
    print(chain)
    nb = NaiveBayes(a,b)
    nb.train()
    nb.process(['stupid','garbage'])
    nb.process(['love','my','dalmation'])

    dateSet, labels = generateDataSet()
    nb2 = NaiveBayes(dateSet,labels)
    nb2.train(is_random=True)
    nb2.test()
