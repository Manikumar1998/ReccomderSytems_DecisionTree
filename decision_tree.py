import numpy as np
from collections import Counter
import utils

class Node(object):
    def __init__(self, feature, value, left=None, right=None, _class=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self._class = _class

    def __repr__(self):
        node_repr = "Node(f_index:{} - value:{} - class:{})"
        reprs = node_repr.format(str(self.feature),
                         str(self.value),
                         str(self._class))
        return reprs

def split(dataset, f_index, value):
    left = {"X":[], "Y": []}
    right = {"X": [], "Y": []}
    X = dataset["X"]
    Y = dataset["Y"]
    
    for x,y in zip(X,Y):
        if x[f_index] == '0':
            left["X"].append(x)
            left["Y"].append(y)
            
        else:
            right["X"].append(x)
            right["Y"].append(y)
    groups = {'left':left, 'right':right}
    return groups

def get_entropy_of_split(dataset, f_index, value, classes):
    groups_dict = split(dataset, f_index, value)
    groups = []
    groups.append(groups_dict['left'])
    groups.append(groups_dict['right'])

    entropy = 0
    total_size = len(dataset['Y'])
    
    for group in groups:
        class_count = Counter(group['Y'])
        length_group = float(len(group["X"]))
        normal_size = float(length_group)/total_size
        group_entropy = 0
    
        for _class in classes:
            if not class_count.has_key(_class):
                continue
            count = class_count[_class]
            prob = float(count)/length_group
            group_entropy -= prob*np.log2(prob)
        entropy += normal_size*group_entropy
    return entropy, groups_dict

def cal_gain(dataset, classes, f_index, value):
    classes_count = Counter(dataset['Y'])
    total_count = len(dataset['Y'])
    initial_entropy = 0

    for _class in classes:
        prob = float(classes_count[_class])/total_count
        initial_entropy -= prob*np.log2(prob)

    entropy_of_split, groups = get_entropy_of_split(dataset, f_index, value, classes)
    gain = initial_entropy-entropy_of_split
    return gain, groups

def majority_voting(dataset):
    Y = dataset['Y']
    Y_count = Counter(Y)
    _class = max(Y_count)
    return _class

def construct_decision_tree(dataset, classes, features, depth, depth_limit, randomtree=False):
    #Stop when depth is reached
    if depth_limit:
        if depth == depth_limit:
            _class = majority_voting(dataset)
            return Node(None,
                        None,
                        _class=_class)

    #Stop when no samples left
    if not dataset['X']:
        return None
        
    #Stop when all belong to same class                
    if len(classes) == 1:
        return Node(None,
                    None,
                    _class=classes[0])

    #Stop when no features are left
    if not features:
        _class = majority_voting(dataset)
        return Node(None,
                    None,
                    _class=_class)
        
    X = dataset['X']
    Y = dataset['Y']

    _gains = []
    for f_index in features:
        information_gain = []

        value = None
        gain, groups = cal_gain(dataset, classes, f_index, value)
        information_gain.append({'value':value,
                                 'gain': gain,
                                 'groups':groups})
            
        max_gain_pair = max(information_gain, key=lambda i:i['gain'])
        _gains.append({'f_index':f_index,
                       'max_gain':max_gain_pair['gain'],
                       'value':max_gain_pair['value'],
                       'groups':max_gain_pair['groups']})

    if not randomtree:
        selected = max(_gains, key=lambda i:i['max_gain'])

    else:
        sorted(_gains, key=lambda i:i['max_gain'])
        top_length = int(np.sqrt(len(features)))
        selected = _gains[random.randint(0, top_length)]
        
    node = Node(selected['f_index'],
                selected['value'],
                _class=None)        
    
    new_features = []
    for f_index in features:
        if f_index != selected['f_index']:
            new_features.append(f_index)

    left_data = selected['groups']['left']
    left_classes = utils.get_classes(left_data)
    
    right_data = selected['groups']['right']
    right_classes = utils.get_classes(right_data)
    
    node.left =  construct_decision_tree(left_data,
                                         left_classes,
                                         new_features,
                                         depth+1,
                                         depth_limit)

    node.right = construct_decision_tree(right_data,
                                         right_classes,
                                         new_features,
                                         depth+1,
                                         depth_limit)
    return node
