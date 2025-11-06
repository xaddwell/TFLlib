import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

class MIA_metrics():
    def __init__(self, y_pred, labels, atk_labels):
        '''
        params:
            y_pred: np.array
            labels: np.array
            atk_labels: np.array
        '''
        self.functions = {
            "acc": self.acc,
            "precision_recall": self.precision_recall,
            'auc':self.auc
            }

        self.y_pred = y_pred
        self.labels = labels
        self.MIA_labels = atk_labels
        self.n_class = np.unique(labels)
        self.ret = {}

    def run(self, config):
        for key, value in config.items():
            self.functions.get(key)(value)
        return self.ret

    def precision_recall(self, option):
        precision, recall, _, _ = precision_recall_fscore_support(y_pred = self.y_pred, y_true = self.MIA_labels, average = "macro")
        self.ret['precision'] = precision
        self.ret['recall'] = recall
        if option == 'per_class':
            self.precision_recall_per_class()
        return
    
    def auc(self, option):
        auc_score = roc_auc_score(self.MIA_labels, self.y_pred)
        self.ret['auc'] = auc_score
        return 

    def acc(self, option):
        acc = accuracy_score(y_pred = self.y_pred, y_true = self.MIA_labels)
        self.ret['acc'] = acc
        if option == 'per_class':
            self.acc_per_class()
        return
    
    def precision_recall_per_class(self):
        precision_per_class, recall_per_class = [], []
        for i in self.n_class:
            c_index = np.where(self.labels == i)[0]
            precision, recall, _, _ = precision_recall_fscore_support(y_pred=self.y_pred[c_index], y_true=self.MIA_labels[c_index])
            precision_per_class.append(precision)
            recall_per_class.append(recall)
        self.ret['precision_per_class'] = precision_per_class
        self.ret['recall_per_class'] = recall_per_class
        return 
    
    def acc_per_class(self):
        accuracy_per_class = []
        for i in self.n_class:
            c_index = np.where(self.labels == i)[0]
            accuracy = accuracy_score(y_pred=self.y_pred[c_index], y_true=self.MIA_labels[c_index])
            accuracy_per_class.append(accuracy)     
        self.ret['accuracy_per_class'] = accuracy_per_class
        return 