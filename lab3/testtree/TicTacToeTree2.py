import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Wczytanie danych z pliku CSV
df = pd.read_csv("ttt.csv")

# Podział danych na cechy i zmienną wynikową
X = df.drop('result', axis=1)
y = df['result']

# Podział zbioru na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Inicjalizacja ostatecznego klasyfikatora z najlepszą maksymalną głębokością
best_max_depth = 10  # Ustawienie najlepszej głębokości
final_clf = DecisionTreeClassifier(max_depth=best_max_depth)
final_clf.fit(X_train, y_train)

# Przewidywanie na zbiorze testowym
y_pred = final_clf.predict(X_test)

# Obliczanie i wyświetlanie dokładności
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność: {accuracy}')

# Rysowanie wykresu drzewa decyzyjnego i zapis do pliku SVG
plt.figure(figsize=(15, 10))
plot_tree(final_clf, filled=True, feature_names=X.columns, class_names=final_clf.classes_)
# plt.savefig("tree_plot.svg")
plt.show()

# Rysowanie wykresu błędu vs Głębokość drzewa i zapis do pliku SVG
max_depths = range(1, 21)
accuracies = []

for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(max_depths, accuracies, marker='o')
plt.title('Dokładność vs Głębokość Drzewa Decyzyjnego')
plt.xlabel('Głębokość Drzewa')
plt.ylabel('Dokładność')
plt.grid(True)
# plt.savefig("accuracy_vs_depth_plot.svg")
plt.show()
