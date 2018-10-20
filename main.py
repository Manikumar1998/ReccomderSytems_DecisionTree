import numpy as np
import random
import multiprocessing as mp
import utils
from decision_tree import construct_decision_tree as DT

if __name__ == '__main__':
    users_db, items_db = utils.read_data()
    accuracy = []
    for index in xrange(10):#len(users_db)):
        print index + 1, 
        X_train, Y_train, X_test, Y_test = utils.extract_data(users_db[index], items_db, 70)
        dataset = {'X': X_train, 'Y': Y_train}
        classes = utils.get_classes(dataset)
        features = range(len(X_train[0]))

        root = DT(dataset, classes, features, 0, 100)
        Y_pred = utils.classify(root, X_test)
    
        indices = utils.get_recommendations(Y_pred, 100)
        confusion_matrix = utils.construct_confusion_matrix(Y_pred, Y_test, indices)

        precision = utils.calc_precision(confusion_matrix)
        recall = utils.calc_recall(confusion_matrix)
        
        accu = utils.accuracy(Y_pred, Y_test)
        print "Accuracy: {} Precision: {} Recall: {}".format(accu, precision, recall)

    
