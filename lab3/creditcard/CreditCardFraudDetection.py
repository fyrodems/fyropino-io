import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from tkinter import filedialog
import tkinter as tk

# Okno dialogowe do wyboru pliku
root = tk.Tk()
root.withdraw()

# Wybór pliku za pomocą okna dialogowego
file_path = filedialog.askopenfilename(title="Wybierz plik CSV", filetypes=[("Pliki CSV", "*.csv")])

# Wczytanie danych z wybranego pliku
data = pd.read_csv(file_path)

# Podział danych na atrybuty (X) i etykiety (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Podział danych na zbiory treningowy i testowy z użyciem funkcji train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
# train_test_split dzieli dane na zbiór treningowy i testowy
# X_train: cechy zbioru treningowego
# X_test: cechy zbioru testowego
# y_train: etykiety zbioru treningowego
# y_test: etykiety zbioru testowego
# test_size=0.2: 20% danych zostanie użyte jako zbiór testowy

# Przygotowanie drzewa decyzyjnego
max_depth_values = range(1, 11)  # Zakres głębokości drzewa do przetestowania
best_depth = 0 # Lista do przechowywania dokładności dla różnych głębokości
best_accuracy = 0
accuracies = []

for max_depth in max_depth_values:
    # Trenowanie drzewa decyzyjnego na zestawie treningowym
    clf = DecisionTreeClassifier(random_state=100, max_depth=max_depth)
    clf.fit(X_train, y_train)

    # Predykcja na zestawie testowym
    y_pred = clf.predict(X_test)

    # Ocena skuteczności modelu
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Aktualizacja najlepszej głębokości i dokładności
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = max_depth

    # Printowanie wyników dla każdej głębokości
    print(f'Głębokość: {max_depth}, Dokładność: {accuracy}')

# Wypisz najlepszą głębokość i dokładność po zakończeniu pętli
print(f'\nNajlepsza głębokość: {best_depth}, Najlepsza dokładność: {best_accuracy}')

# Rysowanie wykresu głębokości drzewa vs dokładność
plt.plot(max_depth_values, accuracies, marker='o')
plt.title('Głębokość drzewa vs dokładność')
plt.xlabel('Głębokość drzewa')
plt.ylabel('Dokładność')

# Zapisywanie wykresu do pliku SVG
plt.savefig('tree_accuracy_plot.svg')

# Rysowanie ostatecznego drzewa decyzyjnego dla najlepszej głębokości
best_clf = DecisionTreeClassifier(random_state=100, max_depth=best_depth)
best_clf.fit(X_train, y_train)
plt.figure(figsize=(25, 10))
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=['0', '1'])

# Zapisywanie drzewa do pliku SVG
plt.savefig('decision_tree.svg')

# Wyświetlanie wykresów
plt.show()
