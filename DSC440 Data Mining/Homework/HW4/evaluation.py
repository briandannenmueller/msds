# for exercise 8.12
# calculate evaluation measures and plot ROC curves

import numpy as np
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, roc_curve

y = np.array([1, 0, 1, 1, 0, 1, 0, 0, 0, 1])
scores = np.array([0.95, 0.85, 0.78, 0.66, 0.60, 0.55, 0.53, 0.52, 0.51, 0.40])
y_hat = np.zeros(len(y))

	
# fpr, tpr, and roc curve
print('---- roc curve ---')
fpr, tpr, thres = roc_curve(y, scores, pos_label=1)
for i in range(len(thres)):
	print('when threshold is ', thres[i])
	print('FPR, TPR = ', fpr[i], tpr[i])
pyplot.plot(fpr, tpr, linestyle='--')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.show()

# calculate confusion metrix
print('---- confusion metrix ----')
for t in thres:
	y_hat = np.where(scores >= t, 1, 0)
	tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
	print('threshold is:', thres)
	print('TN, FP, FN, TP = ',tn, fp, fn, tp)


	
