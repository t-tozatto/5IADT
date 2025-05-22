import numpy as np
import pandas as pd

################## EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações gerais para gráficos
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

df = pd.read_csv('health_insurance_dataset.csv')

numeric_vars = [
    'age', 'income', 'num_dependents_kids',
    'num_dependents_adult', 'num_dependents_older', 'charges'
]

bool_vars = [
    'smoker'
]

categorical_vars = [
    'sex', 'region', 'exercise_frequency', 'bmi', 'marital_status',
    'chronic_condition', 'occupation_risk', 'diet_quality', 'education',
    'alcohol_consumption', 'coverage_level', 'genetic_risk', 'use_last_year'
]

target = 'charges'
random_state = 42

# Visualizar as primeiras linhas
print(df.head())

# Informações gerais sobre o dataset
print(df.info())

# Estatísticas descritivas para variáveis numéricas
print(df.describe())

# Estatísticas para variáveis booleanas e categóricas
print(df.describe(include=['object', 'bool']))



### Análise das variáveis numéricas
for col in numeric_vars:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribuição de {col}')

    sns.boxplot(x=df[col], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'Boxplot de {col}')

    plt.show()

# Matriz
corr = df[numeric_vars].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre Variáveis Numéricas')
plt.show()

### Análise das variáveis booleanas
for col in bool_vars:
    counts = df[col].value_counts(normalize=True) * 100
    print(f'\nDistribuição de {col} (%):\n{counts}')

    sns.countplot(x=col, data=df)
    plt.title(f'Distribuição da variável booleana {col}')
    plt.show()

### Análise das variáveis categóricas
for col in categorical_vars:
    print(f'\nContagem e porcentagem para {col}:')
    print(df[col].value_counts())
    print(df[col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

    plt.figure(figsize=(10, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='pastel')
    plt.title(f'Distribuição da variável categórica {col}')
    plt.show()

### Variáveis vs target
for col in numeric_vars:
    if col != target:
        sns.scatterplot(x=df[col], y=df[target])
        plt.title(f'Relação entre {col} e {target}')
        plt.show()

for col in categorical_vars + bool_vars:
    plt.figure(figsize=(10,5))
    sns.boxplot(x=df[col], y=df[target])
    plt.title(f'{target} por categorias de {col}')
    plt.show()

### Valores faltantes
missing = df.isnull().sum()
print('Valores faltantes por coluna:')
print(missing[missing > 0])

# Outliers nas variáveis numéricas (usando IQR)
for col in numeric_vars:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f'{col}: {len(outliers)} outliers detectados')


################## Pipeline de Pré-processamento
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

if target in numeric_vars:
    numeric_vars.remove(target)

# Pipeline para variáveis numéricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para variáveis booleanas
bool_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Pipeline para variáveis categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_vars),
    ('bool', bool_transformer, bool_vars),
    ('cat', categorical_transformer, categorical_vars)
])



################## Grid Search com Modelos
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, shuffle=True)

# Pipeline com pré-processador (definido anteriormente) e regressor placeholder
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=random_state))
])

# Parâmetros para Grid Search incluindo XGBoost
param_grid = [
    {
        'regressor': [RandomForestRegressor(random_state=random_state)],
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 5, 10]
    },
    {
        'regressor': [GradientBoostingRegressor(random_state=random_state)],
        'regressor__learning_rate': [0.1, 0.05],
        'regressor__n_estimators': [100, 200]
    },
    {
        'regressor': [SVR()],
        'regressor__C': [0.1, 1, 10],
        'regressor__kernel': ['linear', 'rbf']
    },
    {
        'regressor': [xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state, n_jobs=-1)],
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.1, 0.05],
        'regressor__subsample': [0.8, 1],
        'regressor__colsample_bytree': [0.8, 1]
    }
]

# Criar o GridSearchCV com validação cruzada
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=2,
    refit=True
)

## fit
grid_search.fit(X_train, y_train)

## achando o melhor modelo
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)



################## Métricas de validação
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

print("Melhores parâmetros:", grid_search.best_params_)

# # Cross-validation
#from sklearn.model_selection import cross_val_score
#
# scoring = {
#     'MAE': 'neg_mean_absolute_error',
#     'RMSE': 'neg_root_mean_squared_error',
#     'R2': 'r2'
# }
#
# cv_results = cross_val_score(
#     best_model, X, y,
#     cv=5,
#     scoring=scoring,
#     n_jobs=-1
# )
#
# print(f"MAE médio: {-cv_results['test_MAE'].mean():.3f}")
# print(f"RMSE médio: {-cv_results['test_RMSE'].mean():.3f}")
# print(f"R² médio: {cv_results['test_R2'].mean():.3f}")

from sklearn.model_selection import cross_validate

scoring = {
    'MAE': 'neg_mean_absolute_error',
    'RMSE': 'neg_root_mean_squared_error',
    'R2': 'r2'
}

cv_results = cross_validate(
    best_model, X, y,
    cv=5,
    scoring=scoring,
    n_jobs=-1
)

print(f"MAE médio: {-cv_results['test_MAE'].mean():.3f} ± {cv_results['test_MAE'].std():.3f}")
print(f"RMSE médio: {-cv_results['test_RMSE'].mean():.3f} ± {cv_results['test_RMSE'].std():.3f}")
print(f"R² médio: {cv_results['test_R2'].mean():.3f} ± {cv_results['test_R2'].std():.3f}")


################## Validação Final
# Importância das features (para modelos tree-based)
import matplotlib.pyplot as plt
import seaborn as sns

predictions = y_pred

# 1. Importância das features (para modelos tree-based)
regressor = best_model.named_steps['regressor']

if hasattr(regressor, 'feature_importances_'):
    try:
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    except AttributeError:
        feature_names = best_model.named_steps['preprocessor'].get_feature_names()

    importances = regressor.feature_importances_

    # Criar gráfico horizontal
    plt.figure(figsize=(10, 8))
    sns.barplot(x=importances, y=feature_names, palette='viridis')
    plt.title('Importância das Features')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
else:
    print("O modelo não possui atributo 'feature_importances_' para mostrar importância das features.")

# 2. Gráfico de resíduos
residuals = y_test - predictions

plt.figure(figsize=(8, 6))
sns.scatterplot(x=predictions, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Análise de Resíduos')
plt.xlabel('Valores Preditos')
plt.ylabel('Resíduos (Erro)')
plt.tight_layout()
plt.show()


