import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Wczytaj dane z pliku CSV
data = pd.read_csv('CreditCardFraudDetection.csv')
# data = pd.read_csv('creditcard_2023.csv')

# Podziel dane na atrybuty (X) i etykiety (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Podziel dane na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Inicjalizuj drzewo decyzyjne
max_depth_values = range(1, 21)  # Zakres głębokości drzewa do przetestowania
accuracies = []

for max_depth in max_depth_values:
    # Trenuj drzewo decyzyjne na zestawie treningowym
    clf = DecisionTreeClassifier(random_state=100, max_depth=max_depth)
    clf.fit(X_train, y_train)

    # Dokonaj predykcji na zestawie testowym
    y_pred = clf.predict(X_test)

    # Oceń skuteczność modelu
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Wydrukuj wyniki dla każdej głębokości
    print(f'Max Depth: {max_depth}, Accuracy: {accuracy}')

# Rysuj wykres głębokości drzewa vs. dokładność
plt.plot(max_depth_values, accuracies, marker='o')
plt.title('Depth vs. Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.show()
