import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# Wczytaj dane
df = pd.read_csv("ttt.csv")

# Podział na zbiór treningowy i testowy
(train_set, test_set) = train_test_split(df.values, train_size=0.1, random_state=10)
# (train_set, test_set) = train_test_split(df.values, train_size=0.66, random_state=40)


# Podział cech i etykiet
X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
y = train_set[:, [9]]

# Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xtest_scaled = scaler.transform(test_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]])

# Selekcja cech
k_best = 5  # Wybierz odpowiednią liczbę cech
selector = SelectKBest(f_classif, k=k_best)
X_selected = selector.fit_transform(X_scaled, y.ravel())
Xtest_selected = selector.transform(Xtest_scaled)

# Użyj RandomForestClassifier
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_selected, y.ravel())

# Najlepsze parametry z Grid Search
best_params = grid_search.best_params_

# Dopasowanie modelu z najlepszymi parametrami
clf = RandomForestClassifier(**best_params)
clf = clf.fit(X_selected, y.ravel())

# Ocena modelu
accuracy = clf.score(Xtest_selected, test_set[:, [9]])
print("Dokładność modelu:", accuracy)

# Predykcje i macierz pomyłek
predictions = clf.predict(Xtest_selected)
print("Macierz pomyłek:")
print(confusion_matrix(test_set[:, [9]], predictions))
