import matplotlib.pyplot as plt
#import trees

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType,ax1):
    ax1.annotate(nodeTxt,
                 xy=parentPt,
                 xycoords = 'axes fraction',
                 xytext = centerPt,
                 textcoords="axes fraction",
                 va = "center",
                 ha = "center",
                 bbox = nodeType,
                 arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode(createPlot.ax1,"node",(0.5,0.1),(0.1,0.5), decisionNode)
    plotNode(createPlot.ax1,"leafnode",(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

def plotMidText(self,cntrPt, parentPt, txtString,ax1):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    ax1.text(xMid,yMid,txtString)


if __name__=="__main__":
    createPlot()