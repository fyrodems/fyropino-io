import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

df = pd.read_csv("ttt.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.6,
                                         random_state=100)

X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
y = train_set[:, [9]]

clf = GaussianNB()
clf = clf.fit(X, y.ravel())

Xtest = test_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
Ytest = test_set[:, [9]]
print(clf.score(Xtest, Ytest))

predictions = clf.predict(Xtest)

print(confusion_matrix(test_set[:, [9]], predictions))
