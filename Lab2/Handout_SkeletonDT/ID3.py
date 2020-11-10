from collections import Counter
from graphviz import Digraph
import numpy as np
import math


def get_label(target, classes):
    label = classes[0]
    old_count = 0
    for c in classes:
        count = np.sum(target == c)
        if count > old_count:
            old_count = count
            label = c
    return label


def entropy(target, classes):
    p = [np.divide(np.sum([t == c for t in target]), len(target)) for c in classes]
    entropy = -np.sum([pi * math.log(pi, 2) for pi in p if pi > 0])
    return entropy


class ID3DecisionTreeClassifier:

    def __init__(self, minSamplesLeaf=1, minSamplesSplit=2):

        self.__nodeCounter = 0

        # the graph to visualise the tree
        self.__dot = Digraph(comment='The Decision Tree')

        # suggested attributes of the classifier to handle training parameters
        self.__minSamplesLeaf = minSamplesLeaf
        self.__minSamplesSplit = minSamplesSplit

    # Create a new node in the tree with the suggested attributes for the visualisation.
    # It can later be added to the graph with the respective function
    def new_ID3_node(self):
        node = {'id': self.__nodeCounter, 'value': None, 'label': None, 'attribute': None, 'entropy': None,
                'samples': None,
                'classCounts': None, 'nodes': None}

        self.__nodeCounter += 1
        return node

    # adds the node into the graph for visualisation (creates a dot-node)
    def add_node_to_graph(self, node, parentid=-1):
        nodeString = ''
        for k in node:
            if ((node[k] != None) and (k != 'nodes')):
                nodeString += "\n" + str(k) + ": " + str(node[k])

        self.__dot.node(str(node['id']), label=nodeString)
        if (parentid != -1):
            self.__dot.edge(str(parentid), str(node['id']))
            nodeString += "\n" + str(parentid) + " -> " + str(node['id'])

        # print(nodeString)

        return

    # make the visualisation available
    def make_dot_data(self):
        return self.__dot

    # For you to fill in; Suggested function to find the best attribute to split with, given the set of
    # remaining attributes, the currently evaluated data and target.
    def find_split_attr(self, data, target, attributes, classes):
        # Change this to make some more sense
        original_entropy = entropy(target, classes)
        best_information_gain = 0
        best_attribute = None
        too_few_in_leaf = False
        for i, attribute in enumerate(attributes):
            new_entropy_weighted = 0
            for val in attributes[attribute]:
                idx = data[:, i] == val
                if 0 < np.sum(idx) < self.__minSamplesLeaf:
                    too_few_in_leaf = True
                new_entropy_weighted += entropy(target[idx], classes) * np.sum(idx) / len(data)
            information_gain = original_entropy - new_entropy_weighted
            if (information_gain > best_information_gain) & (too_few_in_leaf is False):
                best_information_gain = information_gain
                best_attribute = attribute
            too_few_in_leaf = False
        return best_attribute

    # Calculate the entropy in the set defined by data

    # the entry point for the recursive ID3-algorithm, you need to fill in the calls to your recursive implementation
    def fit(self, data, target, attributes, classes):
        # fill in something more sensible here... root should become the output of the recursive tree creation
        data = np.array(data)
        target = np.array(target)
        root = self.new_ID3_node()
        if root['id'] == 0:
            self.attributes = attributes
        root.update({'entropy': entropy(target, classes),
                     'samples': len(data),
                     'classCounts': [np.sum(target == c) for c in classes],
                     'label': get_label(target, classes),
                     'nodes': []})
        if data.shape[0] < self.__minSamplesSplit:
            self.add_node_to_graph(root)
            return root
        split_attribute = self.find_split_attr(data, target, attributes, classes)
        if split_attribute is not None:
            root.update({'attribute': split_attribute})
            attribute_index = list(attributes).index(split_attribute)
            self.add_node_to_graph(root)
            for val in attributes[split_attribute]:
                idx = data[:, attribute_index] == val
                if np.sum(idx) >= self.__minSamplesLeaf:
                    sub_data = data[idx, :]
                    sub_data = np.delete(sub_data, attribute_index, -1)
                    sub_target = target[idx]
                    sub_attributes = attributes.copy()
                    sub_attributes.pop(split_attribute)
                    child_node = self.fit(sub_data, sub_target, sub_attributes, classes)
                    child_node.update({'value': val})
                    root.update({'nodes': root['nodes'] + [child_node]})
                    self.add_node_to_graph(child_node, root['id'])
        return root

    def predict_one(self, data, tree):
        # fill in something more sensible here... root should become the output of the recursive tree creation
        if tree['attribute'] is None:
            return tree['label']
        attribute_index = list(self.attributes).index(tree['attribute'])
        val = data[attribute_index]
        for child in tree['nodes']:
            if val == child['value']:
                return self.predict_one(data, child)
        return tree['label']


    def predict(self, data, tree):
        predicted = [self.predict_one(row, tree) for row in data]
        return predicted
