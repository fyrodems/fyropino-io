import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix

df = pd.read_csv("diabetes.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
                                         random_state=30)

X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7]]
y = train_set[:, [8]]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

Xtest = test_set[:, [0, 1, 2, 3, 4, 5, 6, 7]]
Ytest = test_set[:, [8]]
print(clf.score(Xtest, Ytest))

predictions = clf.predict(Xtest)

print(confusion_matrix(test_set[:, [8]], predictions))
