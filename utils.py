import random
import matplotlib.pyplot as plt
from decision_tree import LeafNode

def read_data():
    items_data = []
    data = []
    
    with open('datasets/u.data', 'r') as user_fp:
        for line in user_fp:
            data.append(line.strip().split())

    user_ids = sorted(list(set([int(usr[0]) for usr in data])))
    users_data = [[] for _ in xrange(len(user_ids))]

    for user in data:
        users_data[int(user[0])-1].append({'item_id':int(user[1]), 'rating':int(user[2])})

    with open('datasets/u.item', 'r') as item_fp:
        for line in item_fp:
            item = line.strip().split('|')
            ratings = map(int, item[-19:])
            items_data.append({"name": item[1], "genres":ratings})
            
    return users_data, items_data

def extract_data(user_db, items_db, ratio):
    split_len = len(user_db)*ratio/100
    training_set = []
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
        
    while(len(training_set) < split_len):
        index = random.randint(0, len(user_db)-1)
        training_set.append(user_db.pop(index))

    for data in training_set:
        item_id = data['item_id']
        rating  = data['rating']

        X_train.append(items_db[item_id - 1]['genres'])
        Y_train.append(rating)
        
    for data in user_db:
        item_id = data['item_id']
        rating  = data['rating']

        X_test.append(items_db[item_id - 1]['genres'])
        Y_test.append(rating)

    return X_train, Y_train, X_test, Y_test

def get_classes(dataset):
    Y = dataset['Y']
    return list(set(Y))

def accuracy(pred, actual):
    acc = 0.0
    for y_, y in zip(pred, actual):
        if y_ == y:
           acc += 1
    return acc/len(pred)

def classify(root, X_data):
    classified = []
    
    for x in X_data:    
        node = root
        while(node != None):
            if isinstance(node, LeafNode):
                classified.append(node._class)
                break
            else:
                if x[node.feature] == 0:
                    node = node.left
                else:
                    node = node.right
    return classified

def get_recommendations(predictions, k):
    index_pred_pair = []
    for index, pred in enumerate(predictions):
        index_pred_pair.append((index, pred))

    index_pred_pair.sort(reverse=True, key = lambda i:i[1])
    indices = [i[0] for i in index_pred_pair[:k]]
    return indices

def compute_metrics(pred, actual, top_indices):
    tp, fp, all_good = 0, 0, 0
    indices = range(len(pred))
    
    for i in top_indices:
        if pred[i] >= 3:
            if actual[i] >= 3:
                tp += 1
            else:
                fp += 1
    for i in indices:
        if actual[i] >= 3:
            all_good += 1

    precision = calc_precision(tp, fp)
    recall = calc_recall(tp, all_good)
    return precision, recall

def calc_precision(tp, fp):
    try:
        return float(tp)/(tp + fp)
    except:
        return 1.0

def calc_recall(tp, all_good):
    try:
        return float(tp)/all_good
    except:
        return 1.0
        
def calc_MAE(pred, actual):
    n = len(pred)
    assert len(pred) == len(actual)
    sum = 0 
    for p, r in zip(pred, actual):
        sum += abs(p - r)

    res = float(sum)/n
    return res

def calc_RMSE(pred, actual):
    n = len(pred)
    assert len(pred) == len(actual)
    sum = 0
    for p, r in zip(pred, actual):
        sum += (p - r)**2
    
    res = (float(sum)/n)**0.5
    return res            

def plot_graph(precision_dict, recall_dict):
    n = len(precision_dict.keys())
    K = []
    avg_precision = []
    avg_recall = []
    for i in xrange(1, n+1):
        K.append(i)
        avg_precision.append(float(sum(precision_dict[i]))/len(precision_dict[i]))
        avg_recall.append(float(sum(recall_dict[i]))/len(recall_dict[i]))

    plt.xlabel('K')
    plt.ylabel('Avg value')
    plt.title('Precision and Recall')
    precision_plot, = plt.plot(K, avg_precision, 'r', label='Average Precision')
    recall_plot, = plt.plot(K, avg_recall, 'g', label = 'Average Recall')
    plt.legend(handles=[precision_plot, recall_plot])
    plt.show()

def write_to_file(accuracy_arr):
    fp = open('users_accuracy.txt', 'a')
    for i in xrange(len(accuracy_arr)):
        fp.write('User: {} - Test accuracy: {}\n'.format(i + 1, accuracy_arr[i]))
    fp.close()
