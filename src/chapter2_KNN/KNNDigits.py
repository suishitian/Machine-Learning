from kNN import KNN
import os
from numpy import *

def img2vector(imgname,size1,size2):
    returnVect = zeros((1,1024))
    fr = open(imgname)
    for i in range(size1):
        lineStr = fr.readline()
        for j in range(size2):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def getDigitsData(foldername):
    hwlabels = []
    trainingFileList = os.listdir(foldername)
    m = len(trainingFileList)
    trainData = zeros((m,1024))
    for i in range(m):
        filename = trainingFileList[i]
        current_labels = int(filename.split('_')[0])
        hwlabels.append(current_labels)
        trainData[i,:] = img2vector(os.path.join(foldername,filename),32,32)
    return trainData,hwlabels

if __name__ == "__main__":
    errCount = 0
    trainData,trainLabels = getDigitsData("./data/trainingDigits")
    testData,testLabels = getDigitsData("./data/testDigits")
    knn = KNN(3,trainData,trainLabels)
    for i in range(testData.shape[0]):
        flag, result = knn.process(testData[i,:])
        if result != testLabels[i]:
            errCount += 1
            print(str(result) + " is wrong , should be "+str(testLabels[i]))
        else:
            print(str(result)+" is correct")

    print("err is "+str(float(errCount)/(testData.shape[0])))


