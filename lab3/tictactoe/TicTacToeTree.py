import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from tkinter import filedialog
import tkinter as tk

# Okno dialogowe do wyboru pliku
root = tk.Tk()
root.withdraw()  # Ukrycie głównego okna

# Wybór pliku za pomocą okna dialogowego
file_path = filedialog.askopenfilename(title="Wybierz plik CSV", filetypes=[("Pliki CSV", "*.csv")])

# Wczytanie danych z wybranego pliku
df = pd.read_csv(file_path)

# Podział danych na cechy (X) i etykiety (y)
X = df.drop('result', axis=1)
y = df['result']

# Podział danych na zbiory treningowy i testowy z użyciem funkcji train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
# train_test_split dzieli dane na zbiór treningowy i testowy
# X_train: cechy zbioru treningowego
# X_test: cechy zbioru testowego
# y_train: etykiety zbioru treningowego
# y_test: etykiety zbioru testowego
# test_size=0.2: 20% danych zostanie użyte jako zbiór testowy

# Ustalenie optymalnej głębokości drzewa decyzyjnego
final_clf = DecisionTreeClassifier(max_depth=10, random_state=100)
# DecisionTreeClassifier: klasa do tworzenia modelu drzewa decyzyjnego
# max_depth=best_max_depth: ustala maksymalną głębokość drzewa, aby uniknąć przetrenowania

# Trenowanie ostatecznego modelu na zbiorze treningowym
final_clf.fit(X_train, y_train)
# fit(): trenuje model na danych treningowych
# X_train: cechy zbioru treningowego
# y_train: etykiety zbioru treningowego

# Predykcja na zbiorze testowym
y_pred = final_clf.predict(X_test)
# predict(): dokonuje predykcji na danych testowych
# X_test: cechy zbioru testowego

# Obliczenie i wyświetlenie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy}')
# accuracy_score(): oblicza dokładność modelu
# y_test: rzeczywiste etykiety
# y_pred: przewidziane etykiety

# Obliczenie i wyświetlenie macierzy pomyłek
conf_matrix = confusion_matrix(y_test, y_pred)
print("Macierz pomyłek:")
print(conf_matrix)
# confusion_matrix(): tworzy macierz pomyłek
# y_test: rzeczywiste etykiety
# y_pred: przewidziane etykiety

# Wyświetlenie raportu klasyfikacyjnego
print("\nRaport klasyfikacyjny:")
print(classification_report(y_test, y_pred))
# classification_report(): tworzy raport klasyfikacyjny
# y_test: rzeczywiste etykiety
# y_pred: przewidziane etykiety

# Rysowanie drzewa decyzyjnego
plt.figure(figsize=(15, 10))
plot_tree(final_clf, filled=True, feature_names=X.columns, class_names=final_clf.classes_)
plt.title('Drzewo Decyzyjne')
# plot_tree(): rysuje drzewo decyzyjne
# filled=True: koloruje węzły drzewa
# feature_names=X.columns: nazwy cech
# class_names=final_clf.classes_: nazwy klas

# Ewaluacja dokładności w zależności od głębokości drzewa decyzyjnego
max_depths = range(1, 16)
accuracies = []

# Dla każdej głębokości drzewa, trenuj model, dokonaj predykcji i oblicz dokładność
for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Rysowanie wykresu dokładności w zależności od głębokości drzewa decyzyjnego
plt.figure(figsize=(10, 6))
plt.plot(max_depths, accuracies, marker='o')
# plot(): rysuje wykres
# max_depths: lista głębokości drzewa
# accuracies: lista dokładności dla każdej głębokości
# marker='o': oznacza punkty na wykresie

plt.title('Dokładność vs Głębokość Drzewa Decyzyjnego')
plt.xlabel('Głębokość Drzewa')
plt.ylabel('Dokładność')
plt.grid(True)
# Zapisanie wykresu do pliku
plt.savefig("accuracy_vs_depth_plot.svg")
plt.show()

# Oczekujemy zobaczenia, jak dokładność modelu zmienia się wraz ze wzrostem głębokości drzewa.
# Optymalna głębokość powinna być taka, która daje dobrą dokładność bez przetrenowania modelu.

# Patrząc na wykres, możemy zauważyć, że punkt, w którym dokładność jest odpowiednio wysoka i nie rośnie dalej równa się 10
# Z tego względu w klasie DecisionTreeClassifier ustawiono max_depth na 10


