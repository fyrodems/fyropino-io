import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

df = pd.read_csv("diabetes.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=30)

featureNames = df.columns.values[:8]
classNames = ["tested_positive", "tested_negative"]

print(featureNames)

X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7]]
y = train_set[:, [8]]

Xtest = test_set[:, [0, 1, 2, 3, 4, 5, 6, 7]]
Ytest = test_set[:, [8]]

depths = range(1, 21)  # Przykładowe głębokości drzewa od 1 do 10
accuracies = []

for depth in depths:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X, y)

    predictions = clf.predict(Xtest)
    accuracy = accuracy_score(Ytest, predictions)
    accuracies.append(accuracy)

    print(f"Accuracy for max_depth={depth}: {accuracy}")

# Rysowanie wykresu dokładności w zależności od głębokości drzewa
plt.figure(figsize=(10, 6))
plt.plot(depths, accuracies, marker='o')
plt.title('Accuracy vs. Max Depth of Decision Tree')
plt.xlabel('Max Depth of Tree')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Pozostała część kodu do rysowania drzewa
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X, y)

predictions = clf.predict(Xtest)

# print(confusion_matrix(Ytest, predictions))

fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(clf,
                   feature_names=featureNames,
                   class_names=classNames,
                   filled=True)

fig.savefig("diab-decision_tree.svg")
