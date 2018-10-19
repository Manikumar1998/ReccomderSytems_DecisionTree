import random

def read_data():
    items_data = []
    data = []
    
    with open('datasets/u.data', 'r') as user_fp:
        for line in user_fp:
            data.append(line.strip().split())

    user_ids = sorted(list(set([int(usr[0]) for usr in data])))
    users_data = [[] for _ in xrange(len(user_ids))]

    for user in data:
        users_data[int(user[0])-1].append({'item_id':int(user[1]), 'rating':user[2]})

    with open('datasets/u.item', 'r') as item_fp:
        for line in item_fp:
            item = line.strip().split('|')
            ratings = item[-19:]
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
            if node._class != None:
                classified.append(node._class)
                break
            else:
                if x[node.feature] == '0':
                    node = node.left
                else:
                    node = node.right
    return classified
