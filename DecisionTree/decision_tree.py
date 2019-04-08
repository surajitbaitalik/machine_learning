# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.metrics import confusion_matrix
import pydotplus
def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    part_dict = {val: (x == val).nonzero()[0] for val in np.unique(x)}

    return part_dict
    raise Exception('Function not yet implemented!')


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    val, count = np.unique(y, return_counts=True)
    p = count.astype('float') / len(y)
    entropy = 0.0
    entropy=- np.sum(p * np.log2(p))
    return entropy
    # INSERT YOUR CODE HERE
    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    valx, countx = np.unique(x, return_counts=True)
    px = countx.astype('float') / len(x)
    hyx = 0.0
    for pxval, xval in zip(px, valx):
        hyx += (pxval) * entropy(y[x == xval])
    hy = entropy(y)
    ixy = hy - hyx
    return ixy


    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    root={}
    if attribute_value_pairs is None:
        attribute_value_pairs = np.vstack([[(i, val) for val in np.unique(x[:, i])] for i in range(x.shape[1])])
    yval,ycount=np.unique(y, return_counts=True)
    #1
    if len(yval)==1:
        return yval[0]
    #2 & #3
    if len(attribute_value_pairs)==0 or depth==max_depth:
        return yval[np.argmax(ycount)]

    list_mutualinformation = np.array([mutual_information(np.array(x[:, i] == v).astype(int), y) for (i, v) in attribute_value_pairs])
    (bestattr, bestvalue) = attribute_value_pairs[np.argmax(list_mutualinformation)]

    # Based on best attribute and value, splitting in true or false
    partitions = partition(np.array(x[:, bestattr] == bestvalue).astype(int))

    # Removing the best attribute and value to split on
    dropindex = np.all(attribute_value_pairs == (bestattr, bestvalue), axis=1)
    attribute_value_pairs = np.delete(attribute_value_pairs, np.argwhere(dropindex), 0)
    # Removing those values that were split on  and recursively calling with the new data.
    for splitindex, index in partitions.items():
        xsubset = x.take(index, axis=0)
        ysubset = y.take(index, axis=0)
        decision = bool(splitindex)

        root[(bestattr, bestvalue, decision)] = id3(xsubset, ysubset, attribute_value_pairs=attribute_value_pairs,
                                                    max_depth=max_depth, depth=depth + 1)

    return root

    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    for splitval, sub_tree in tree.items():
        attr_index = splitval[0]
        attr_value = splitval[1]
        split_decision = splitval[2]

        if split_decision == (x[attr_index] == attr_value):
            if type(sub_tree) is dict:
                label = predict_example(x, sub_tree)
            else:
                label = sub_tree

            return label

    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    n = len(y_true)
    err = [y_true[i] != y_pred[i] for i in range(n)]
    return sum(err) / n
    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training data
    M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Learn a decision tree of depth 3
    decision_tree = id3(Xtrn, ytrn, max_depth=3)

    # Pretty print it to console
    pretty_print(decision_tree)

    # Visualize the tree and save it as a PNG image
    dot_str = to_graphviz(decision_tree)
    render_dot_file(dot_str, './my_learned_tree')

    # Compute the test error
    y_pred = [predict_example(x, decision_tree) for x in Xtst]
    tst_err = compute_error(ytst, y_pred)

    print('Test Error of Monk-1 problem = {0:4.2f}%.'.format(tst_err * 100))

    #compute the training error
    y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
    trn_err = compute_error(ytrn, y_pred_trn)
    print('Training  Error of Monk-1 problem= {0:4.2f}%.'.format(trn_err * 100))

    # Question (b)((Learning Curves),plots for three data set(here i didn't repeat the code for 3 data types,
    # instead taking separately and providing plots in report)
    #load train data
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    test_error = []
    train_error = []
    for i in range(1, 11):
        decision_tree = id3(Xtrn, ytrn, max_depth=i)
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)
        test_error.append(tst_err * 100)
        y_pred_trn = [predict_example(x, decision_tree) for x in Xtrn]
        trn_err = compute_error(ytrn, y_pred_trn)
        train_error.append(trn_err * 100)
    print('test error list contents>>>>>')
    print(test_error)
    print('training  error list contents>>>>>')
    print(train_error)

    xaxistest = [i for i in range(1, 11)]
    yaxistest = test_error
    xaxistrn = [i for i in range(1, 11)]
    yaxistrn = train_error
    plt.plot(xaxistest, yaxistest, linewidth=1.0)
    plt.plot(xaxistrn, yaxistrn, linewidth=1.0)
    plt.xlabel('depth')
    plt.ylabel('Test error in blue and train error in red')
    plt.title("Monks-3 Training Testing error vs depth of the tree")
    plt.show()


   #C.(Weak Learners) learn decision tree and scikit learn's confusion matrix



    Mtrn = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    Mtst = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = Mtrn[:, 0]
    Xtrn = Mtrn[:, 1:]
    ytst = Mtst[:, 0]
    Xtst = Mtst[:, 1:]
    for i in range(1,6,2):
        decision_tree = id3(Xtrn, ytrn, max_depth=i)

        # Pretty print it to console
        pretty_print(decision_tree)

        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(decision_tree)
        render_dot_file(dot_str, './my_learned_tree_depth_{}'.format(i))
        # Compute the test error
        y_pred = [predict_example(x, decision_tree) for x in Xtst]
        tst_err = compute_error(ytst, y_pred)
        matrix = confusion_matrix(ytst, y_pred)
        print("Confusion Matrix for depth_{}".format(i))
        print(matrix)
        print('Accuracy of Monk-1 problem = {0:4.2f}%.'.format((1-tst_err) * 100))





     #d.scikit-learn entropy criterion
    col_head = ['y', 'x1', 'x2', 'x3', 'x4', 'x5']
    Mtrn = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    Mtst = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = Mtrn[:, 0]
    Xtrn = Mtrn[:, 1:]
    ytst = Mtst[:, 0]
    Xtst = Mtst[:, 1:]
    for i in range(1, 6, 2):
        clf = DecisionTreeClassifier(max_depth=i,criterion='entropy')
        clf = clf.fit(Xtrn, ytrn)
        y_pred = clf.predict(Xtst)
        print("Accuracy for depth>>{}:".format(i), metrics.accuracy_score(ytst, y_pred)*100)
        matrix = confusion_matrix(ytst, y_pred)
        print("Confusion Matrix of scikit-learn's for depth>>{}".format(i))
        print(matrix)
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=col_head, class_names=['0', '1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('mydecision_scikit_entropy{}.png'.format(i))

        Image(graph.create_png())


     #e. other data set


    # with scikit-learns and  entropy
    col_head = ['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15',
                'x16', 'x17', 'x18', 'x19', 'x20']
    Mtrn = np.genfromtxt('./mushroom.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    Mtst = np.genfromtxt('./mushroom.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = Mtrn[:, 0]
    Xtrn = Mtrn[:, 1:]
    ytst = Mtst[:, 0]
    Xtst = Mtst[:, 1:]
    for i in range(1, 6, 2):
        clf = DecisionTreeClassifier(max_depth=i,criterion='entropy')
        clf = clf.fit(Xtrn, ytrn)
        y_pred = clf.predict(Xtst)
        print("Accuracy for depth>>{}:".format(i), metrics.accuracy_score(ytst, y_pred))
        matrix = confusion_matrix(ytst, y_pred)
        print("Confusion Matrix for depth>>{}".format(i))
        print(matrix)
        dot_data = StringIO()
        export_graphviz(clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=col_head, class_names=['0', '1'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('mydecision_scikit_mushroom_entropy{}.png'.format(i))

        Image(graph.create_png())

        # my learned tree

        for i in range(1, 6, 2):
            decision_tree = id3(Xtrn, ytrn, max_depth=i)

            # Pretty print it to console
            pretty_print(decision_tree)

            # Visualize the tree and save it as a PNG image
            dot_str = to_graphviz(decision_tree)
            render_dot_file(dot_str, './my_learned_tree_mushroom_{}'.format(i))
            # Compute the test error
            y_pred = [predict_example(x, decision_tree) for x in Xtst]
            tst_err = compute_error(ytst, y_pred)
            matrix = confusion_matrix(ytst, y_pred)
            print("Confusion Matrix for mushroom_depth_{}".format(i))
            print(matrix)
            print('Accuracy of Mushroom problem = {0:4.2f}%.'.format((1 - tst_err) * 100))



