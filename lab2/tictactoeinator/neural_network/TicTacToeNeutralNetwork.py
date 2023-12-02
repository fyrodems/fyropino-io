import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from tkinter import filedialog
import tkinter as tk

# # Okno dialogowe do wyboru pliku
# root = tk.Tk()
# root.withdraw()  # Ukrycie głównego okna
#
# # Wybór pliku za pomocą okna dialogowego
# file_path = filedialog.askopenfilename(title="Wybierz plik CSV", filetypes=[("Pliki CSV", "*.csv")])
#
# # Wczytanie danych z wybranego pliku
# df = pd.read_csv(file_path)

# Wczytaj dane
df = pd.read_csv("../ttt.csv")

# Podział na zbiór treningowy i testowy
(train_set, test_set) = train_test_split(df.values, train_size=0.6, random_state=100)

# Wydzielenie danych wejściowych (X) i wyjściowych (y) z zbioru treningowego
X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
y = train_set[:, [9]]

# Utworzenie i dopasowanie modelu MLP
# Utworzenie instancji MLPClassifier
# mlp = MLPClassifier(hidden_layer_sizes=(99, 90, 81, 72, 63, 54, 45, 36, 27, 18, 9), max_iter=120)
# mlp = MLPClassifier(hidden_layer_sizes=(9, 18, 27, 36), max_iter=1200)
# mlp = MLPClassifier(hidden_layer_sizes=(2, 4, 8, 16, 32, 64), max_iter=100)
# mlp = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=479, alpha=0.001, learning_rate='adaptive')
# mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=479)
mlp = MLPClassifier(hidden_layer_sizes=(200,), max_iter=3000)

# Dopasowanie modelu do danych treningowych
mlp.fit(X, y.ravel())  # ravel() do spłaszczenia listy

# Ocena dokładności na zbiorze testowym
Xtest = test_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
Ytest = test_set[:, [9]]
print(mlp.score(Xtest, Ytest))

# Predykcje na zbiorze testowym i wyświetlenie macierzy błędów
predictions = mlp.predict(Xtest)
print(confusion_matrix(Ytest, predictions))


