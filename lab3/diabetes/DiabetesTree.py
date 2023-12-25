import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv(file_path)
    df_cleaned = df.dropna()

    class_mapping = {'tested_negative': 0, 'tested_positive': 1}
    df_cleaned['class'] = df_cleaned['class'].map(class_mapping)

    return df_cleaned

def oversample_data(df):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(df.iloc[:, :-1], df['class'])
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    return df_resampled

def split_data(df_resampled):
    train_set, test_set = train_test_split(df_resampled, train_size=0.66, random_state=42)
    return train_set, test_set

def preprocess_data(train_set, test_set):
    feature_names = train_set.columns.values[:8]
    class_names = ["tested_negative", "tested_positive"]

    X_train = train_set.iloc[:, :8]
    y_train = train_set.iloc[:, 8]

    X_test = test_set.iloc[:, :8]
    y_test = test_set.iloc[:, 8]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, feature_names, class_names

def optimize_model(X_train_scaled, y_train):
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    return grid_search.best_estimator_

def evaluate_model(model, X_test_scaled, y_test, class_names):
    accuracy = model.score(X_test_scaled, y_test)
    print(f'Dokładność modelu: {accuracy}')

    predictions = model.predict(X_test_scaled)
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Macierz pomyłek:")
    print(conf_matrix)

    print("Raport klasyfikacji:")
    print(classification_report(y_test, predictions, target_names=class_names))

    return accuracy

def main():
    file_path = "diabetes.csv"
    df_cleaned = load_data(file_path)
    df_resampled = oversample_data(df_cleaned)
    train_set, test_set = split_data(df_resampled)
    X_train_scaled, y_train, X_test_scaled, y_test, feature_names, class_names = preprocess_data(train_set, test_set)

    model = optimize_model(X_train_scaled, y_train)
    accuracy = evaluate_model(model, X_test_scaled, y_test, class_names)

    print(f'Ostateczna dokładność modelu: {accuracy}')

if __name__ == "__main__":
    main()
