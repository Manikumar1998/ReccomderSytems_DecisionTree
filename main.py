import numpy as np
import random
import multiprocessing as mp
import utils
from decision_tree import construct_decision_tree as DT

if __name__ == '__main__':
    users_db, items_db = utils.read_data()
    accuracy = []
    for index in xrange(len(users_db)):
        print index + 1, 
        X_train, Y_train, X_test, Y_test = utils.extract_data(users_db[index], items_db, 70)
        dataset = {'X': X_train, 'Y': Y_train}
        classes = utils.get_classes(dataset)
        features = range(len(X_train[0]))

        root = DT(dataset, classes, features, 0, 100)
        Y_ = utils.classify(root, X_test)
        accu = utils.accuracy(Y_, Y_test)
        print "Accuracy: {}".format(accu)
        accuracy.append(accu)
    accuracy.sort(reverse=True)
    print accuracy


    
