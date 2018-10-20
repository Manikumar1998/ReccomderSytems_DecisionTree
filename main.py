import numpy as np
import random
import multiprocessing as mp
import utils
from decision_tree import construct_decision_tree as DT
from timeit import default_timer as timer

if __name__ == '__main__':
    users_db, items_db = utils.read_data()
    MAE_arr, RMSE_arr, accuracy_arr = [], [], []
    precision_dict, recall_dict = {}, {}

    K = 100

    for k in xrange(1, K+1):
        precision_dict[k] = []
        recall_dict[k] = []

    depth = int(raw_input("Enter the maximum depth of the tree (0 for no limit): "))
    
    start = timer()
    for index in xrange(len(users_db)):
        X_train, Y_train, X_test, Y_test = utils.extract_data(users_db[index], items_db, 70)
        dataset = {'X': X_train, 'Y': Y_train}
        
        classes = utils.get_classes(dataset)
        features = range(len(X_train[0]))

        root = DT(dataset, classes, features, 0, depth)
        Y_pred = utils.classify(root, X_test)

        for k in xrange(1, K+1):
            if k <= len(Y_test):
                top_K_indices = utils.get_recommendations(Y_pred, k)
                precision, recall = utils.compute_metrics(Y_pred, Y_test, top_K_indices)
                precision_dict[k].append(precision)
                recall_dict[k].append(recall)

        MAE = utils.calc_MAE(Y_pred, Y_test)
        RMSE = utils.calc_RMSE(Y_pred, Y_test)
        accu = utils.accuracy(Y_pred, Y_test)
        
        MAE_arr.append(MAE)
        RMSE_arr.append(RMSE)
        accuracy_arr.append(accu)
        print "User: {} - Test accuracy: {}".format(index+1, accu)

    print "\nTime taken: {} sec".format(timer() - start)
    print 'Average MAE: {}'.format(sum(MAE_arr)/float(len(MAE_arr)))
    print 'Average RMSE: {}'.format(sum(RMSE_arr)/float(len(RMSE_arr)))
    print 'Maximum test accuracy: {}'.format(max(accuracy_arr))
    print 'Minimum test accuracy: {}'.format(min(accuracy_arr))
    print 'Average Accuracy: {}'.format(sum(accuracy_arr)/float(len(accuracy_arr)))
    utils.write_to_file(accuracy_arr)
    utils.plot_graph(precision_dict, recall_dict)
