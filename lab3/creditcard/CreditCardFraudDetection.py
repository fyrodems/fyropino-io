import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# Wczytaj dane z pliku CSV
data = pd.read_csv('CreditCardFraudDetection.csv')

# Podziel dane na atrybuty (X) i etykiety (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Podziel dane na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Inicjalizuj drzewo decyzyjne
max_depth_values = range(1, 11)  # Zakres głębokości drzewa do przetestowania
best_depth = 0
best_accuracy = 0
accuracies = []

for max_depth in max_depth_values:
    # Trenuj drzewo decyzyjne na zestawie treningowym
    clf = DecisionTreeClassifier(random_state=100, max_depth=max_depth)
    clf.fit(X_train, y_train)

    # Dokonaj predykcji na zestawie testowym
    y_pred = clf.predict(X_test)

    # Oceń skuteczność modelu
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Aktualizuj najlepszą głębokość i dokładność
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = max_depth

    # Wydrukuj wyniki dla każdej głębokości
    print(f'Max Depth: {max_depth}, Accuracy: {accuracy}')

# Wypisz najlepszą głębokość i dokładność po zakończeniu pętli
print(f'\nBest Depth: {best_depth}, Best Accuracy: {best_accuracy}')

# Rysuj wykres głębokości drzewa vs. dokładność
plt.plot(max_depth_values, accuracies, marker='o')
plt.title('Depth vs. Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')

# Zapisz wykres do pliku SVG
plt.savefig('depth_vs_accuracy_plot.svg')

# Rysuj drzewo decyzyjne o najlepszej głębokości
best_clf = DecisionTreeClassifier(random_state=100, max_depth=best_depth)
best_clf.fit(X_train, y_train)
plt.figure(figsize=(25, 10))
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=['0', '1'])

# Zapisz drzewo do pliku SVG
plt.savefig('decision_tree.svg')

# Pokaż wykresy
plt.show()
