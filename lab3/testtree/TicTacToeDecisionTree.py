# import numpy
# import pandas as pd
# import graphviz
# from sklearn.model_selection import train_test_split
# from sklearn import tree
# from sklearn.metrics import confusion_matrix
# from matplotlib import pyplot as plt
#
# df = pd.read_csv("ttt.csv")
# (train_set, test_set) = train_test_split(df.values, train_size=0.66,
#                                          random_state=45)
#
# X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
# y = train_set[:, [9]]
#
# clf = tree.DecisionTreeClassifier(max_depth=5)
# clf = clf.fit(X, y)
# dot_data = tree.export_graphviz(clf, out_file=None)
#
# tree.plot_tree(clf)
#
# graph = graphviz.Source(dot_data)
# graph.render("tictactoe")
#
# Xtest = test_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
# Ytest = test_set[:, [9]]
# XtestFeatures = ["pole11", "pole13", "pole13", "pole21", "pole22", "pole23", "pole31", "pole32", "pole33"]
# Xfeatures = numpy.array(XtestFeatures)
# targetNames = ["wygrana", "remis"]
# print(clf.score(Xtest, Ytest))
#
# predictions = clf.predict(Xtest)
#
# print(confusion_matrix(test_set[:, [9]], predictions))
#
# text_representation = tree.export_text(clf)
# print(text_representation)
#
#
# fig = plt.figure(figsize=(25, 20))
# _ = tree.plot_tree(clf,
#                    feature_names=XtestFeatures,
#                    class_names=targetNames,
#                    filled=True)
#
# fig.savefig("decision_tree.png")
