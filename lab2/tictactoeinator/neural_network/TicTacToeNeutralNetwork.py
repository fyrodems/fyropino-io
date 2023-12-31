import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from tkinter import filedialog
import tkinter as tk

# Poniżej jest drugi skrypt, który odpali program 50 razy
# Zlicza średnią dokładność i ilość wyników powyżej i poniżej 90%
# W celach testowych można zakomentować powyższy kod i odkomentować poniższy

# Okno dialogowe do wyboru pliku
root = tk.Tk()
root.withdraw()  # Ukrycie głównego okna

# Wybór pliku za pomocą okna dialogowego
file_path = filedialog.askopenfilename(title="Wybierz plik CSV", filetypes=[("Pliki CSV", "*.csv")])

# Wczytanie danych z wybranego pliku
df = pd.read_csv(file_path)

# Tutaj zacząć komentarz

# Podział na zbiór treningowy i testowy
(train_set, test_set) = train_test_split(df.values, train_size=0.6, random_state=100)

# Wydzielenie danych wejściowych (X) i wyjściowych (y) z zbioru treningowego
X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
y = train_set[:, [9]]

# Utworzenie i dopasowanie modelu MLP (Multi-layer Perceptron)
# Utworzenie instancji MLPClassifier z jedną warstwą ukrytą (hidden layer) zawierającą 200 neuronów
# max_iter określa maksymalną liczbę iteracji (epok), aby zapobiec zbyt długiemu trenowaniu
mlp = MLPClassifier(hidden_layer_sizes=(200,), max_iter=3000)

# Dopasowanie modelu do danych treningowych
mlp.fit(X, y.ravel())  # ravel() do spłaszczenia listy

# Ocena dokładności na zbiorze testowym
Xtest = test_set[:, :9]
Ytest = test_set[:, [9]]
print(f'Dokładność na zbiorze testowym: {mlp.score(Xtest, Ytest)}')

# Predykcje na zbiorze testowym i wyświetlenie macierzy błędów
predictions = mlp.predict(Xtest)
print(f'Macierz błędów:\n{confusion_matrix(Ytest, predictions)}')

# Tutaj skończyć komentarz i odkomentować poniższy kod

# # Utwórz plik do logów
# log_file_path = "TicTacToeNeutralNetwork_logs.txt"
# with open(log_file_path, "w") as log_file:
#     log_file.write("Logi z uruchomienia skryptu\n\n")
#
# # Inicjalizacja listy na wyniki dokładności
# accuracy_list = []
#
# # Inicjalizacja liczników
# above_90_count = 0
# below_90_count = 0
#
# # Powtórz 30 razy
# for i in range(50):
#     # Podział na zbiór treningowy i testowy
#     (train_set, test_set) = train_test_split(df.values, train_size=0.6, random_state=100)
#
#     # Wydzielenie danych wejściowych (X) i wyjściowych (y) z zbioru treningowego
#     X = train_set[:, :9]
#     y = train_set[:, 9]
#
#     # Utworzenie i dopasowanie modelu MLP
#     mlp = MLPClassifier(hidden_layer_sizes=(200,), max_iter=4000)
#     mlp.fit(X, y.ravel())
#
#     # Ocena dokładności na zbiorze testowym
#     Xtest = test_set[:, :9]
#     Ytest = test_set[:, 9]
#     score = mlp.score(Xtest, Ytest)
#
#     # Dodanie wyników dokładności do listy
#     accuracy_list.append(score)
#
#     # Zliczenie wyników powyżej i poniżej 90%
#     if score > 0.9:
#         above_90_count += 1
#     else:
#         below_90_count += 1
#
#     # Predykcje na zbiorze testowym i wyświetlenie macierzy błędów
#     predictions = mlp.predict(Xtest)
#     confusion_matrix_ = confusion_matrix(Ytest, predictions)
#
#     # Wypisanie wyników w konsoli
#     print(f"Iteracja {i + 1}:")
#     print(f"Dokładność: {score}")
#     print(f"Macierz błędów:\n{confusion_matrix_}")
#     print("-" * 30)
#
#     # Logowanie wyników do pliku
#     with open(log_file_path, "a") as log_file:
#         log_file.write(f"Iteracja {i + 1}:\n")
#         log_file.write(f"Dokładność: {score}\n")
#         log_file.write(f"Macierz błędów:\n{confusion_matrix_}\n")
#         log_file.write("-" * 30 + "\n")
#
# # Obliczenie średniej dokładności
# average_accuracy = sum(accuracy_list) / len(accuracy_list)
#
# # Podsumowanie
# summary = f"\nPodsumowanie:\nIlość iteracji: {len(accuracy_list)}\nIlość wyników powyżej 90%: {above_90_count}\nIlość wyników poniżej 90%: {below_90_count}\nŚrednia dokładność: {average_accuracy}"
# print(summary)
#
# # Zapisanie podsumowania do pliku
# with open(log_file_path, "a") as log_file:
#     log_file.write(summary)
