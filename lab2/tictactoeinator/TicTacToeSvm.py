from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tkinter import filedialog
import tkinter as tk

# Okno dialogowe do wyboru pliku
root = tk.Tk()
root.withdraw()

# Wybór pliku za pomocą okna dialogowego
file_path = filedialog.askopenfilename(title="Wybierz plik CSV", filetypes=[("Pliki CSV", "*.csv")])

# Wczytanie danych z wybranego pliku
df = pd.read_csv(file_path)

# Przekształcenie etykiet na liczby
df['result'] = df['result'].map({'positive': 1, 'negative': 0})

# Podział danych na zbiór treningowy i testowy
train_set, test_set = train_test_split(df, train_size=0.66, random_state=100)

# Przygotowanie danych treningowych
X_train = train_set.iloc[:, :9].values
y_train = train_set.iloc[:, 9].values

# Przygotowanie danych testowych
X_test = test_set.iloc[:, :9].values
y_test = test_set.iloc[:, 9].values

# Inicjalizacja i konfiguracja modelu SVM
model = SVC(
    C=1.0,                   # Parametr kosztu, kontroluje wagę dla błędów klasyfikacyjnych.
    kernel='rbf',            # Wybór jądra (kernel) - 'rbf' oznacza radialną funkcję bazową.
    gamma='scale',           # Współczynnik gamma w jądrze ('scale' oznacza 1 / (liczba cech * wariancja danych)).
    random_state=100          # Ziarno losowości dla powtarzalności.
)

# Trenowanie modelu
model.fit(X_train, y_train)

# Testowanie modelu na danych testowych
predictions = model.predict(X_test)

# Wypisanie przewidywań
for i in range(len(X_test)):
    print(f"Wejścia: {X_test[i]}, Przewidziane: {'TAK' if predictions[i] == 1 else 'NIE'}")

# Wypisanie dokładności modelu na danych testowych
accuracy_test = accuracy_score(y_test, predictions)
print(f"Dokładność modelu na danych testowych: {accuracy_test}")
