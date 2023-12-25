# Importowanie niezbędnych bibliotek
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Wczytanie danych z pliku CSV
df = pd.read_csv('hatespeech.csv')

# Przetwarzanie tekstu - wybór kolumny 'tweet' jako danych wejściowych
X = df['tweet']
y = df['class']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
# Komentarz 1: Dane zostają podzielone na zbiór treningowy i testowy w proporcji 80-20.

# Przygotowanie danych do modelu za pomocą CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
# Komentarz 2: Tekst jest przekształcany na reprezentację liczbową za pomocą CountVectorizer.

# Inicjalizacja listy głębokości drzewa do sprawdzenia
depths = range(1, 51)
accuracies = []  # Lista do przechowywania dokładności dla różnych głębokości
best_depth = None
best_accuracy = 0.0

# Pętla po różnych głębokościach drzewa
for depth in depths:
    # Tworzenie modelu drzewa decyzyjnego
    model = DecisionTreeClassifier(max_depth=depth)
    # Trenowanie modelu na danych treningowych
    model.fit(X_train_vectorized, y_train)
    # Predykcja na danych testowych
    y_pred = model.predict(X_test_vectorized)
    # Obliczanie dokładności modelu
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    # Komentarz 3: Model drzewa decyzyjnego jest trenowany i oceniany na danych testowych.
    print(f'Głębokość drzewa: {depth}, Dokładność: {accuracy}')

    # Aktualizacja najlepszej głębokości i dokładności
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth

# Rysowanie wykresu głębokości drzewa vs. dokładność
plt.figure(figsize=(20, 10))
plt.plot(depths, accuracies, marker='o')
plt.title('Głębokość drzewa vs. Dokładność')
plt.xlabel('Głębokość drzewa')
plt.ylabel('Dokładność')
plt.show()
# Komentarz 4: Wykres przedstawia zależność dokładności modelu od głębokości drzewa.

# Wypisanie najlepszej głębokości drzewa i odpowiadającej dokładności
print(f"Najlepsza głębokość drzewa: {best_depth}, Najlepsza dokładność: {best_accuracy}")

# Rysowanie drzewa decyzyjnego dla najlepszej głębokości
best_model = DecisionTreeClassifier(max_depth=best_depth)
best_model.fit(X_train_vectorized, y_train)

plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, feature_names=vectorizer.get_feature_names_out())
plt.show()
# Komentarz 5: Rysowanie ostatecznego drzewa decyzyjnego dla najlepszej głębokości.
