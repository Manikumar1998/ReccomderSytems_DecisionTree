import math

def precision(list_items_classify,list_items_test):
    count = 0;
    for i in list_items_classify:
        for j in list_items_test:
            if (i==j):
                count += 1
    precision = count/len(list_items_classify)
    return precision

def recall(list_items_classify,list_items_test):
    count = 0
    for i in list_items_classify:
        for j in list_items_test:
            if (i==j):
                count += 1
    recall = count/len(list_items_test)
    return recall

def MAE(list_items_classify,list_items_test,Y_,Y_test):
    count = 0
    sum_mae =0
    for i in list_items_classify:
        for j in list_items_test:
            if (i==j):
                count += 1
                index1 = list_items_classify.index(i)
                index2 = list_items_test.index(j)
                value = abs(Y_[index1] - Y_test[index2])
                sum_mae += value
    MAE_value = sum_mae/count
    return MAE_value

def RMSE(list_items_classify,list_items_test,Y_,Y_test):
    count = 0
    sum_RMSE =0
    for i in list_items_classify:
        for j in list_items_test:
            if (i==j):
                count += 1
                index1 = list_items_classify.index(i)
                index2 = list_items_test.index(j)
                value = math.pow(Y_[index1] - Y_test[index2],2)
                sum_RMSE += value
    RMSE_value = sum_RMSE/count
    RMSE_value = math.sqrt(RMSE_value)
    return RMSE_value




                
