import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Wczytanie danych
df = pd.read_csv("diabetes.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.66, random_state=30)

# Wybór cech i etykiet
featureNames = df.columns.values[:8]
classNames = ["tested_positive", "tested_negative"]

X = train_set[:, :8]
y = train_set[:, 8]

Xtest = test_set[:, :8]
Ytest = test_set[:, 8]

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xtest_scaled = scaler.transform(Xtest)

# Inżynieria cech (dodatkowy przykład: sumaryczna masa ciała i wiek)
X_scaled = pd.DataFrame(X_scaled, columns=featureNames)
X_scaled['bmi_times_age'] = X_scaled['mass'] * X_scaled['age']

# Balans klas (przykładowa waga dla klasy "tested_negative")
class_weights = {'tested_positive': 2, 'tested_negative': 3}

# Tuning parametrów i kross-walidacja
rf_clf = RandomForestClassifier(n_estimators=100, random_state=30, class_weight=class_weights)

# Dopasowanie modelu
rf_clf.fit(X_scaled, y)

# Kross-walidacja z 5 podziałami po dopasowaniu modelu
cv_scores = cross_val_score(rf_clf, X_scaled, y, cv=5, scoring='accuracy')

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean()}")

# Badanie istotności cech po dopasowaniu modelu
feature_importances = rf_clf.feature_importances_
print("Feature Importances:")
for feature, importance in zip(X_scaled.columns, feature_importances):
    print(f"{feature}: {importance}")

# Dopasowanie ostatecznego modelu i ewaluacja na zbiorze testowym
Xtest_scaled = pd.DataFrame(Xtest_scaled, columns=featureNames)
Xtest_scaled['bmi_times_age'] = Xtest_scaled['mass'] * Xtest_scaled['age']

predictions_rf = rf_clf.predict(Xtest_scaled)
accuracy_rf = accuracy_score(Ytest, predictions_rf)
print(f"Random Forest Accuracy on Test Set: {accuracy_rf}")
