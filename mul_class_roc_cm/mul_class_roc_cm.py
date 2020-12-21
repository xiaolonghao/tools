from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
from numpy import interp
from sklearn.metrics import confusion_matrix
import itertools

pre = model.predict(X_test,batch_size=batch_size)
classes = pre.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pre[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
lw = 2
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pre.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig(dataset_prefix+'roc.jpg')
# plt.show()

predictions = model.predict(X_test, batch_size=batch_size)
predictions = predictions.argmax(axis=-1)
truelabel = y_test.argmax(axis=-1)
cm = confusion_matrix(y_true=truelabel, y_pred=predictions)
plt.figure()

accuracy = np.trace(cm) / float(np.sum(cm))
misclass = 1 - accuracy

plt.figure(figsize=(15, 12))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
plt.title('Confusion Matrix')
plt.colorbar()
thresh = cm.max() / 2
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, "{:,}".format(cm[i, j]),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=30)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
plt.savefig(dataset_prefix+'confusionmatrix.jpg',dpi=350)
plt.show()

