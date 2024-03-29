import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='rgb or flow')
args = parser.parse_args()
num_classes=7
mode = args.mode
data_src = [
    ['i3d', mode, 'f', 'cat'],
    ['i3d', mode, 's', 'cat'],
    ['i3d', mode, 'fs', 'avg'],
    ['i3d', mode, 'fs', 'svm'],
    ['mbi3d', mode, 'fs', 'cat'],
    ['mbi3d', mode, 'fs', 'cbp'],
]

data = []
for e in data_src:
    data.append(np.mean(np.stack([np.load('pev_split_%d_%s_%s_%s_%s_acc.npy' % \
        tuple([i] + e)) for i in range(1,4) ]), axis=0).tolist())

np.set_printoptions(precision=4)
print(np.mean(np.array(data),axis=1))

bw = 0.10  

plt.title('Mean Accuracy', fontsize=20)
label_names = ['Pointing', 'Attention', 'Passing', 'Receiving', 'Positive', 'Negative', 'Gesture']
methods = ['A','B', 'A+B Avg','A+B Svm','A+B Cat','A+B Cbp']
index = range(num_classes)
for e,p in zip(data,range(num_classes)):
    plt.bar([i+p*bw for i in index], e, bw, label=methods[p])
    p += 1

plt.xticks([index + 0.3 for index in range(num_classes)], label_names)
plt.xlabel("Action")
plt.legend(bbox_to_anchor=(0.85, 0.80, 0., 0.), loc=3)
plt.savefig('Acc_%s.eps' % mode,format='eps')
#plt.bar(index, values1, bw)
#plt.bar(index+bw, values2, bw)
#plt.bar(index+2*bw, values3, bw)
#plt.xticks(index+1.5*bw, ['A', 'B', 'C', 'D', 'E'])
#plt.show()
