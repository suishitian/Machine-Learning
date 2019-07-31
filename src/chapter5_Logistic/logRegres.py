def loadDataSet():
    dataMat = []
    label = []
    fr = open('./data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append()