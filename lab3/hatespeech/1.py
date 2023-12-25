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

# Podział danych na zbiory treningowy i testowy z użyciem funkcji train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
# train_test_split dzieli dane na zbiór treningowy i testowy
# X_train: cechy zbioru treningowego
# X_test: cechy zbioru testowego
# y_train: etykiety zbioru treningowego
# y_test: etykiety zbioru testowego
# test_size=0.2: 20% danych zostanie użyte jako zbiór testowy

# Przygotowanie danych do modelu za pomocą CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
# Tekst jest przekształcany na reprezentację liczbową za pomocą CountVectorizer

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
    # Model drzewa decyzyjnego jest trenowany i oceniany na danych testowych
    print(f'Głębokość drzewa: {depth}, Dokładność: {accuracy}')

    # Aktualizacja najlepszej głębokości i dokładności
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth

# Rysowanie wykresu głębokości drzewa vs dokładność
plt.figure(figsize=(20, 10))
plt.plot(depths, accuracies, marker='o')
plt.title('Głębokość drzewa vs dokładność')
plt.xlabel('Głębokość drzewa')
plt.ylabel('Dokładność')
plt.show()
# Wykres przedstawia zależność dokładności modelu od głębokości drzewa

# Ponieważ dla różnych danych optymalna głębokość drzewa była różna,
# drzewo jest przycinane do głębokości repezentowenej przez best_depth

# Wypisanie najlepszej głębokości drzewa i odpowiadającej dokładności
print(f"Najlepsza głębokość drzewa: {best_depth}, Najlepsza dokładność: {best_accuracy}")

# Rysowanie drzewa decyzyjnego dla najlepszej głębokości
best_model = DecisionTreeClassifier(max_depth=best_depth)
best_model.fit(X_train_vectorized, y_train)

# Rysowanie ostatecznego drzewa decyzyjnego dla najlepszej głębokości
plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, feature_names=vectorizer.get_feature_names_out())
plt.show()

