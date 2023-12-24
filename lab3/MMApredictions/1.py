import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Wczytaj plik CSV
df = pd.read_csv('predictions.csv')

# Konwertuj zmienne kategoryczne na wartości liczbowe
df['correct'] = df['correct'].astype(int)

# Wybierz cechy (predyktory)
X = df[['winner_odds']]  # Dodaj więcej cech, jeśli to konieczne

# Wybierz zmienną docelową
y = df['correct']

# Podziel zbiór danych na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Przygotuj listę głębokości drzewa do przetestowania
max_depths = list(range(1, 31))  # Od 1 do 30

# Inicjalizuj listę do przechowywania wyników dokładności
accuracies = []

# Testuj różne głębokości drzewa
for max_depth in max_depths:
    # Utwórz klasyfikator lasu losowego z określoną głębokością drzewa
    model = RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=1)

    # Wytrenuj model
    model.fit(X_train, y_train)

    # Dokonaj predykcji na zestawie testowym
    y_pred = model.predict(X_test)

    # Oblicz dokładność
    accuracy = accuracy_score(y_test, y_pred)

    # Dodaj dokładność do listy
    accuracies.append(accuracy)

# Rysuj wykres
plt.plot(max_depths, accuracies, marker='o')
plt.title('Głębokość Drzewa vs. Dokładność (Random Forest)')
plt.xlabel('Głębokość Drzewa')
plt.ylabel('Dokładność')
plt.grid(True)
plt.show()
