import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Wczytanie danych
df = pd.read_csv("ttt.csv")

# Podział na zbiór treningowy i testowy
(train_set, test_set) = train_test_split(df.values, train_size=0.65, random_state=100)
# (train_set, test_set) = train_test_split(df.values, train_size=0.4, random_state=45)

# Wydzielenie danych wejściowych (X) i wyjściowych (y) z zbioru treningowego
X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
y = train_set[:, [9]]

# Utworzenie i dopasowanie modelu Random Forest
clf = RandomForestClassifier()
clf.fit(X, y)

# Ocena dokładności na zbiorze testowym
Xtest = test_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
Ytest = test_set[:, [9]]
print(clf.score(Xtest, Ytest))

# Predykcje na zbiorze testowym i wyświetlenie macierzy błędów
predictions = clf.predict(Xtest)
print(confusion_matrix(test_set[:, [9]], predictions))
