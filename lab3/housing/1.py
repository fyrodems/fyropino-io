import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

# Wczytywanie danych
df = pd.read_csv("housing_price_dataset.csv")

# Kodowanie kategorii za pomocą LabelEncoder
le = LabelEncoder()
df["Neighborhood"] = le.fit_transform(df["Neighborhood"])

# Decision Tree
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rysowanie wykresu dokładności w zależności od głębokości drzewa
depth_range = range(1, 21)
accuracy_scores = []

for depth in depth_range:
    tree_model = DecisionTreeRegressor(max_depth=depth)
    scores = cross_val_score(tree_model, X_train, y_train, cv=5, scoring='r2')
    accuracy = scores.mean()
    accuracy_scores.append(accuracy)
    print(f"Głębokość: {depth}, Dokładność: {accuracy}")

# Wybieramy optymalną głębokość drzewa na podstawie wykresu
optimal_depth = depth_range[np.argmax(accuracy_scores)]
print(f"\nOptymalna Głębokość Drzewa: {optimal_depth}")

# Uczenie modelu z optymalną głębokością
optimal_tree_model = DecisionTreeRegressor(max_depth=optimal_depth)
optimal_tree_model.fit(X_train, y_train)

# Wizualizacja drzewa i zapis do pliku
plt.figure(figsize=(10, 6))
plot_tree(optimal_tree_model, filled=True, feature_names=X.columns, rounded=True, max_depth=optimal_depth)
plt.title(f'Drzewo Decyzyjne (Głębokość {optimal_depth})')
plt.savefig('decision_tree.png')
plt.close()

# Wykres dokładności i zapis do pliku
plt.figure(figsize=(10, 6))
plt.plot(depth_range, accuracy_scores, marker='o')
plt.title('Dokładność Drzewa Decyzyjnego vs. Głębokość Drzewa')
plt.xlabel('Głębokość Drzewa')
plt.ylabel('Dokładność Drzewa')
plt.grid(True)
plt.savefig('wykres_dokladnosci.png')
plt.close()
