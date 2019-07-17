import matplotlib.pyplot as plt
import trees

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(ax1,nodeTxt, centerPt, parentPt, nodeType):
    ax1.annotate(nodeTxt,
                            xy=parentPt,
                            xycoords = 'axes fraction',
                            xytext = centerPt,
                            textcoords="axes fraction",
                            va = "center",
                            ha = "center",
                            bbox = nodeType,
                            arrowprops=arrow_args
                            )

def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode(createPlot.ax1,"node",(0.5,0.1),(0.1,0.5), decisionNode)
    plotNode(createPlot.ax1,"leafnode",(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

if __name__=="__main__":
    createPlot()