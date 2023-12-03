import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from tkinter import filedialog
import tkinter as tk

# Okno dialogowe do wyboru pliku
root = tk.Tk()
root.withdraw()  # Ukrycie głównego okna

# Wybór pliku za pomocą okna dialogowego
file_path = filedialog.askopenfilename(title="Wybierz plik CSV", filetypes=[("Pliki CSV", "*.csv")])

# Wczytanie danych z wybranego pliku
df = pd.read_csv(file_path)

# Utwórz plik do logów
log_file_path = "TicTacToeNeutralNetwork_logs.txt"
with open(log_file_path, "w") as log_file:
    log_file.write("Logi z uruchomienia skryptu\n\n")

# Inicjalizacja listy na wyniki dokładności
accuracy_list = []

# Inicjalizacja liczników
above_90_count = 0
below_90_count = 0

# Powtórz 30 razy
for i in range(50):
    # Podział na zbiór treningowy i testowy
    (train_set, test_set) = train_test_split(df.values, train_size=0.6, random_state=100)

    # Wydzielenie danych wejściowych (X) i wyjściowych (y) z zbioru treningowego
    X = train_set[:, :9]
    y = train_set[:, 9]

    # Utworzenie i dopasowanie modelu MLP
    mlp = MLPClassifier(hidden_layer_sizes=(200,), max_iter=3000)
    mlp.fit(X, y.ravel())

    # Ocena dokładności na zbiorze testowym
    Xtest = test_set[:, :9]
    Ytest = test_set[:, 9]
    score = mlp.score(Xtest, Ytest)

    # Dodaj wynik dokładności do listy
    accuracy_list.append(score)

    # Zlicz wyniki powyżej i poniżej 90%
    if score > 0.9:
        above_90_count += 1
    else:
        below_90_count += 1

    # Predykcje na zbiorze testowym i wyświetlenie macierzy błędów
    predictions = mlp.predict(Xtest)
    confusion_matrix_ = confusion_matrix(Ytest, predictions)

    # Wypisz wyniki na konsolę
    print(f"Iteracja {i + 1}:")
    print(f"Dokładność: {score}")
    print(f"Macierz błędów:\n{confusion_matrix_}")
    print("-" * 30)

    # Loguj wyniki do pliku
    with open(log_file_path, "a") as log_file:
        log_file.write(f"Iteracja {i + 1}:\n")
        log_file.write(f"Dokładność: {score}\n")
        log_file.write(f"Macierz błędów:\n{confusion_matrix_}\n")
        log_file.write("-" * 30 + "\n")

# Oblicz średnią dokładność
average_accuracy = sum(accuracy_list) / len(accuracy_list)

# Podsumowanie
summary = f"\nPodsumowanie:\nIlość iteracji: {len(accuracy_list)}\nIlość wyników powyżej 90%: {above_90_count}\nIlość wyników poniżej 90%: {below_90_count}\nŚrednia dokładność: {average_accuracy}"
print(summary)

# Zapisz podsumowanie do pliku
with open(log_file_path, "a") as log_file:
    log_file.write(summary)
