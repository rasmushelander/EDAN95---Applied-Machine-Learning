import ToyData as td
import ID3
import graphviz
import numpy as np
from sklearn import tree, metrics, datasets
from sklearn.tree import DecisionTreeClassifier


def main():

    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()

    id3 = ID3.ID3DecisionTreeClassifier(1, 2)

    myTree = id3.fit(data, target, attributes, classes)
    plot = id3.make_dot_data()
    plot.render("testTree")
    graph = graphviz.Source(plot)
    graph.render('test')
    predicted = id3.predict(data2, myTree)

    print(predicted)


if __name__ == "__main__": main()