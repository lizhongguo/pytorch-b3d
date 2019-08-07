import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def draw_confusion_matrix(matrix,labels,color='Oranges'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    x, y = matrix.shape
    
    for i in range(x):
        for j in range(y):
            ax.text(i,j,'%.1f' % matrix[i][j].item(),fontsize=10,horizontalalignment='center',verticalalignment='center')

    cax = ax.matshow(matrix.numpy(),cmap=cm.get_cmap(color))
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks,labels,rotation=-45,horizontalalignment='center')
    plt.yticks(tick_marks,labels,rotation=-45,verticalalignment='center')
    plt.xlabel('GroundTruth')
    plt.ylabel('Predicted')
    #plt.show()
    return fig