import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("Загрузка данных...")
path_to_csv = r"C:\Users\anna_\.cache\kagglehub\datasets\mlg-ulb\creditcardfraud\versions\3\creditcard.csv"
df = pd.read_csv(path_to_csv)

scaler = StandardScaler()
df['Time'] = scaler.fit_transform(df[['Time']])
df['Amount'] = scaler.fit_transform(df[['Amount']])

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Доля мошеннических транзакций в тесте: {y_test.mean():.4f}")

def evaluate_model(model_name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    f1 = f1_score(y_te, y_pred, average='macro')
    print(f"Модель: {model_name} | Macro F1: {f1:.4f}")
    return model, y_pred


# 1. Бэггинг (с балансировкой классов)
bagging_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(class_weight='balanced'),
    n_estimators=50, random_state=42, n_jobs=-1
)
evaluate_model("Bagging", bagging_model, X_train, y_train, X_test, y_test)

# 2. Случайный лес
rf_model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
)
evaluate_model("Random Forest", rf_model, X_train, y_train, X_test, y_test)

# 3. LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=100, is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1
)
evaluate_model("LightGBM", lgbm_model, X_train, y_train, X_test, y_test)

print("\n--- Задание 2: Оптимизация (поиск по сетке) ---")

param_grid_lgbm = {
    'n_estimators': [50, 100, 200],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100]
}

lgbm_search = RandomizedSearchCV(
    LGBMClassifier(is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    param_distributions=param_grid_lgbm,
    n_iter=5, scoring='f1_macro', cv=3, random_state=42, n_jobs=-1
)

print("Обучение оптимизированного LightGBM...")
lgbm_search.fit(X_train, y_train)
best_lgbm = lgbm_search.best_estimator_
y_pred_best_lgbm = best_lgbm.predict(X_test)
f1_lgbm = f1_score(y_test, y_pred_best_lgbm, average='macro')

print(f"Лучшие параметры LightGBM: {lgbm_search.best_params_}")
print(f"Optimized LightGBM Macro F1: {f1_lgbm:.4f}")

print("\nПостроение матрицы ошибок для лучшей модели...")
cm = confusion_matrix(y_test, y_pred_best_lgbm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legal (0)', 'Fraud (1)'])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax)
plt.title('Confusion Matrix - Optimized LightGBM')
plt.show()

# ==========================================
# ЗАДАНИЕ 3. CatBoost
# ==========================================
print("\n--- Задание 3: CatBoost ---")

catboost_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',
    verbose=0,
    random_seed=42
)

catboost_model.fit(X_train, y_train)
y_pred_cat = catboost_model.predict(X_test)
f1_cat = f1_score(y_test, y_pred_cat, average='macro')
print(f"CatBoost Macro F1: {f1_cat:.4f}")

print("\nОптимизация CatBoost (небольшой перебор)...")
param_grid_cat = {
    'iterations': [100, 300],
    'depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1]
}
cat_search = RandomizedSearchCV(
    CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_seed=42),
    param_distributions=param_grid_cat,
    n_iter=3, scoring='f1_macro', cv=3, random_state=42, n_jobs=-1
)
cat_search.fit(X_train, y_train)
print(f"Optimized CatBoost Macro F1: {cat_search.best_score_:.4f} (CV score)")
print(f"Лучшие параметры CatBoost: {cat_search.best_params_}")
