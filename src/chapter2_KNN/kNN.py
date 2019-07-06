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

    def process(self,input):
        dataSetSize = self.dataSet.shape[0]
        diffMat = tile()