import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from itertools import islice

df = pd.read_csv("diabetes.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=30)

featureNames = df.columns.values[:8]
classNames = ["tested_positive", "tested_negative"]

print(featureNames)

X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7]]
y = train_set[:, [8]]

clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X, y)

Xtest = test_set[:, [0, 1, 2, 3, 4, 5, 6, 7]]
Ytest = test_set[:, [8]]
print(clf.score(Xtest, Ytest))

predictions = clf.predict(Xtest)

print(confusion_matrix(test_set[:, [8]], predictions))

fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(clf,
                   feature_names=featureNames,
                   class_names=classNames,
                   filled=True)

fig.savefig("diab-decision_tree.svg")
