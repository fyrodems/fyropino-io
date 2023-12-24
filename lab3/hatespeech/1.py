import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Wczytaj dane z pliku CSV
df = pd.read_csv('hatespeech.csv')

# Przetwarzanie tekstu - można dostosować w zależności od potrzeb
# W tym przypadku korzystamy z kolumny 'tweet' jako danych wejściowych
X = df['tweet']
y = df['class']

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Przygotowanie danych do modelu (można użyć tf-idf, CountVectorizer itp.)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Budowa modelu drzewa decyzyjnego
depths = range(1, 51)  # Zakres głębokości drzewa do sprawdzenia
accuracies = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth)
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print(depth)

# Rysowanie wykresu głębokości drzewa vs. dokładność
plt.plot(depths, accuracies, marker='o')
plt.title('Głębokość drzewa vs. Dokładność')
plt.xlabel('Głębokość drzewa')
plt.ylabel('Dokładność')
plt.show()
