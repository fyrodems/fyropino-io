import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Podział na zbiór treningowy i testowy
# Ustawienie train_size na 0.65 oznacza, że 65% danych zostanie użyte do treningu, a 35% do testowania.
# Ustawienie random_state pozwala na reprodukowalność wyników.
(train_set, test_set) = train_test_split(df.values, train_size=0.65, random_state=100)
# Alternatywny podział danych z innymi wartościami dla train_size i random_state.
# (train_set, test_set) = train_test_split(df.values, train_size=0.4, random_state=45)

# Wydzielenie danych wejściowych (X) i wyjściowych (y) z zbioru treningowego
# Kolumny od 0 do 8 to dane wejściowe, a kolumna 9 to etykiety/wyniki.
X = train_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
y = train_set[:, [9]]

# Utworzenie i dopasowanie modelu Random Forest
# Utworzenie instancji RandomForestClassifier
clf = RandomForestClassifier()
# Dopasowanie modelu do danych treningowych
clf.fit(X, y)

# Ocena dokładności na zbiorze testowym
# Przygotowanie danych testowych
Xtest = test_set[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]]
Ytest = test_set[:, [9]]
# Wypisanie dokładności modelu na zbiorze testowym
print(clf.score(Xtest, Ytest))

# Predykcje na zbiorze testowym i wyświetlenie macierzy błędów
# Uzyskanie predykcji dla danych testowych
predictions = clf.predict(Xtest)
# Wyświetlenie macierzy błędów do oceny jakości modelu
print(confusion_matrix(test_set[:, [9]], predictions))
