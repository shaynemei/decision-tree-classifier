import numpy as np
import os, sys

def load_svm(path, vocab, use_pipe=False):
    with open(path) as f:
        num_entries = 0
        for line in f.readlines():
            num_entries += 1
            if not use_pipe:
                line = line.strip().split()
                vocab |= {feature.split(':')[0] for feature in line[1:]}
    vocab = list(vocab)
    vocab.sort()
    
    with open(path) as f:
        data = np.zeros((num_entries, len(vocab) + 1))
        for i, line in enumerate(f.readlines()):
            line = line.strip().split()
            y = y_dict[line[0]]
            data[i][len(vocab)] = y_dict[line[0]]
            features = {feature.split(':')[0] for feature in line[1:]}
            for j, word in enumerate(vocab):
                if word in features:
                    data[i][j] = 1
    return data, vocab

def entropy(col):
    counts = np.unique(col, return_counts=True)[1]
    probs = np.array([count/len(col) for count in counts])
    ent = -probs.dot(np.log2(probs))
    return ent

def infogain(data, feature):
    x = data[:, feature]
    y = data[:, -1]
    joint_col = np.sum([x, y*2], axis=0)
    h_x = entropy(x)
    h_y = entropy(y)
    h_xy = entropy(joint_col)
    return h_x + h_y - h_xy

def get_best_feature(data):
    max_IG = 0
    best_feature = 0
    for feature in range(0, len(vocab)):
        IG = infogain(data, feature)
        if IG > max_IG:
            max_IG = IG
            best_feature = feature
    return best_feature, max_IG

class DecisionTree:
    """
    A binary decision tree.
    """
    class Node:
        """
        A node object in a binary decision tree.
        """
        def __init__(self, feature=None, label_prob=None, num_instances=None):
            """
            Parameters
            -----------
            feature: the index of the vocab representing the feature that is being split on at this node
            label_prob: the probability of each class at a leaf node
            num_instances: total number of instances at a leaf node
            left: left child pointing to the node that HAS the feature word on this node
            right: right child pointing to the node that does NOT have the feature word on this node
            """
            self.feature = feature
            self.label_prob = label_prob
            self.num_instances = num_instances
            self.left = None
            self.right = None

        def __repr__(self):
            if self.is_leaf():
                return f"{str(self.num_instances)} {str(self.label_prob)}"
            else:
                return str(self.feature)

        def is_leaf(self):
            if self.label_prob is None:
                return False
            else:
                return True
    
    def __init__(self, vocab):
        self.root = self.Node()
        self.vocab = vocab
    
    def fit(self, data, min_gain, max_depth):
        """
        Parameters
        ----------
        data: numpy array with class label at the last column. Class labels are encoded as y_dict.
        min_gain: the minimum information gain threshold to stop growing tree.
        max_depth: the maximum depth of the tree to grow.
        """
        self.root = DecisionTree.split(data, self.root, min_gain, max_depth, 0)
    
    def print_model(self):
        model = []
        DecisionTree.print_model_helper(self.root, "", model)
        return model
        
    def _predict_inst(self, inst, return_prob):
        node = self.root
        while not node.is_leaf():
            if inst[node.feature]:
                node = node.left
            else:
                node = node.right
        if not return_prob:
            return np.argmax(node.label_prob)
        else:
            return node.label_prob
    
    def predict(self, inp, return_prob=False):
        """
        Parameters
        ----------
        inp: a list of instances that in the feature space of vocab
        """
        res = []
        for inst in inp:
            res.append(self._predict_inst(inst, return_prob))
        return res
        
    @staticmethod
    def split(data, node, min_gain, max_depth, depth):

        def _create_leaf(data, node):
            num_instances = len(data)
            node.num_instances = num_instances
            if num_instances == 0:
                node.label_prob = [0, 0, 0]
                return node

            node.label_prob = [len(data[data[:,-1] == 0]) / num_instances,
                               len(data[data[:,-1] == 1]) / num_instances,
                               len(data[data[:,-1] == 2]) / num_instances]
            return node

        depth += 1

        # edge case: no more entries to split
        if len(data) <= 1:
            return _create_leaf(data, node)

        # early stop condition 1: reach max depth
        if depth > max_depth:
            return _create_leaf(data, node)

        best_feature, IG = get_best_feature(data)

        # early stop condition 2: less than min infogain
        if IG <= min_gain:
            return _create_leaf(data, node)

        node.feature = best_feature
        node.left = DecisionTree.split(data[np.where(data[:,best_feature] == 1)], DecisionTree.Node(), min_gain, max_depth, depth)
        node.right = DecisionTree.split(data[np.where(data[:,best_feature] == 0)], DecisionTree.Node(), min_gain, max_depth, depth)
        return node
    
    @staticmethod
    def print_model_helper(node, path, model):
        if node.is_leaf():
            probs = node.label_prob
            path = path[:-2]
            path += (f" {node.num_instances} talk.politics.guns {node.label_prob[0]} talk.politics.mideast {node.label_prob[1]} talk.politics.misc {node.label_prob[2]}")
            model.append(path)
            return
        else:
            DecisionTree.print_model_helper(node.left, path + f"HAS{vocab[node.feature]}->", model)
            DecisionTree.print_model_helper(node.right, path + f"NOT{vocab[node.feature]}->", model)
        return model

    
def print_tree(node, depth=0):
    q = []
    # each node has exactly 0 or 2 children
    if node.is_leaf():
        print("\t" * depth + f"{node}")
    else:
        print("\t" * depth + f"{vocab[node.feature]}")
    if node.right is not None:
        q.append(node.left)
        q.append(node.right)
    if not len(q):
        return
    for node in q:
        print_tree(node, depth + 1)
        
def print_confusion_matrix(res, truth):
    labels = y_dict.keys()
    counts_dict = count_res(res, truth)
    print("                  ", end="")
    for label in labels:
        print(f"{label} ", end="")
    print()
    for i, label in enumerate(labels):
        col0 = counts_dict[i] if i in counts_dict else 0
        col1 = counts_dict[i + 3] if i+3 in counts_dict else 0
        col2 = counts_dict[i + 6] if i+6 in counts_dict else 0
        print(f"{label}\t{col0}\t\t{col1}\t\t{col2}")
        
def print_accuracy(res, truth):
    counts_dict = count_res(res, truth)
    col0 = counts_dict[0] if 0 in counts_dict else 0
    col1 = counts_dict[4] if 4 in counts_dict else 0
    col2 = counts_dict[8] if 8 in counts_dict else 0
    print((col0+col1+col2)/len(truth))

def count_res(res, truth):
    unique, counts = np.unique(np.sum([res, truth*3], axis=0), return_counts=True)
    counts_dict = dict()
    for i, j in zip(unique, counts):
        counts_dict[i] = j
    return counts_dict


# usage: build dt.sh training_data test_data max_depth min_gain model_file sys_output > acc_file
if __name__ == "__main__":
    file_train = sys.argv[1]
    file_test = sys.argv[2]
    max_depth = int(sys.argv[3])
    min_gain = float(sys.argv[4])
    model_out = sys.argv[5]
    sys_out = sys.argv[6]

    y_dict = dict({"talk.politics.guns": 0,
         "talk.politics.mideast": 1,
         "talk.politics.misc": 2})
    vocab = set()
    train, vocab = load_svm(f"{file_train}", vocab)
    test, vocab = load_svm(f"{file_test}", vocab, True)
    dt = DecisionTree(vocab)
    dt.fit(train, min_gain, max_depth)
    model = dt.print_model()
    res_train = dt.predict(train)
    res_test = dt.predict(test)
    truth_train = train[:,-1]
    truth_test = test[:,-1]

    with open(f"{model_out}", 'w') as f:
        for line in model:
            f.write(line)

    with open(f"{sys_out}", 'w') as f:
        f.write("%%%%% training data:\n")
        for i, probs in enumerate(dt.predict(train, return_prob=True)):
            f.write(f"array{i}: talk.politics.guns\t{probs[0]}\ttalk.politics.mideast\t{probs[1]}\ttalk.politics.misc\t{probs[2]}\n")

        f.write("\n")

        f.write("%%%%% test data:\n")
        for i, probs in enumerate(dt.predict(train, return_prob=True)):
            f.write(f"array{i}: talk.politics.guns\t{probs[0]}\ttalk.politics.mideast\t{probs[1]}\ttalk.politics.misc\t{probs[2]}\n")

    print("Confusion matrix for the training data:\nrow is the truth, column is the system output\n\n")
    print_confusion_matrix(res_train, truth_train)
    print()
    print_accuracy(res_train, truth_train)
    print("\n")
    print("Confusion matrix for the testing data:\nrow is the truth, column is the system output\n\n")
    print_confusion_matrix(res_test, truth_test)
    print()
    print_accuracy(res_test, truth_test)






