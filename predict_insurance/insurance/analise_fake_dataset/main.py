# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay

# Configurações
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', 100)
np.random.seed(42)

# ==============================================
# CARREGAMENTO DE DADOS
# ==============================================
df = pd.read_csv('health_insurance_dataset.csv')

# Visualização inicial
print("\n=== Primeiras linhas ===")
print(df.head())
print("\n=== Informações do dataset ===")
print(df.info())
print("\n=== Estatísticas descritivas ===")
print(df.describe().T)

# ==============================================
# ANÁLISE EXPLORATÓRIA (EDA)
# ==============================================
def perform_eda(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df['charges'], kde=True, bins=30)
    plt.title('Distribuição dos Custos de Seguro')
    plt.savefig('charges_distribution.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df, alpha=0.6)
    plt.title('Idade vs Custos por Status de Fumante')
    plt.savefig('age_vs_charges.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='region', y='charges', data=df)
    plt.title('Distribuição de Custos por Região')
    plt.savefig('region_impact.png')
    plt.close()

    plt.figure(figsize=(14, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.savefig('correlation_matrix.png')
    plt.close()

perform_eda(df)

# ==============================================
# ANÁLISE DE DISTRIBUIÇÃO DE CADA PROPRIEDADE
# ==============================================
def plot_feature_distribution(df, numeric_features):
    """
    Função para criar histogramas para cada variável numérica.
    """
    for feature in numeric_features:
        plt.figure(figsize=(12, 6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f'Distribuição de {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequência')
        plt.savefig(f'distribution_{feature}.png')
        plt.close()

def plot_feature_smoker(df):
    """
    Função para criar histogramas para cada variável numérica.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='smoker', data=df, palette='viridis', hue='smoker', dodge=False)

    # Adicionar anotações de contagem e porcentagem
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 50,
                f'{height}\n({height / total:.1%})',
                ha='center', va='bottom', fontsize=10)

    plt.title('Distribuição de Fumantes vs Não-Fumantes', fontsize=14)
    plt.xlabel('Fumante', fontsize=12)
    plt.ylabel('Contagem', fontsize=12)
    plt.xticks([0, 1], ['Não-Fumante', 'Fumante'])  # Labels descritivos
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('smoker_distribution.png', bbox_inches='tight')
    plt.close()

def plot_categorical_distribution(df, categorical_features):
    """
    Função para criar gráficos de barras para cada variável categórica.
    """
    for feature in categorical_features:
        plt.figure(figsize=(12, 6))
        sns.countplot(x=feature, data=df, palette='Set2')
        plt.title(f'Distribuição de {feature}')
        plt.xlabel(feature)
        plt.ylabel('Contagem')
        plt.xticks(rotation=45)  # Para melhorar a visualização dos rótulos, caso necessário
        plt.savefig(f'distribution_{feature}.png')
        plt.close()


# Gerar a análise de distribuição para as variáveis numéricas
numeric_features = [
    'age', 'bmi', 'chronic_conditions', 'genetic_risk',
    'alcohol_consumption', 'income', 'regional_cost_factor', 'coverage_factor',
    'dependents'
]

binary_features = [
    'smoker'
]

categorical_features = [
    'sex', 'region', 'exercise',
    'diet', 'occupation_risk', 'coverage_level'
]

plot_feature_distribution(df, numeric_features)
plot_feature_smoker(df)
plot_categorical_distribution(df, categorical_features)

# ==============================================
# PRÉ-PROCESSAMENTO
# ==============================================

# Verificar valores nulos
print("\n=== Valores Nulos ===")
print(df.isnull().sum())

# Preencher valores numéricos com a mediana e valores categóricos com o modo (valor mais frequente)
imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

# Aplicar imputação nos dados
df[numeric_features] = imputer_num.fit_transform(df[numeric_features])
df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

# Verificar se todos os valores nulos foram tratados
print("\n=== Após imputação ===")
print(df.isnull().sum())

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder())
        ]), categorical_features)
    ])

X = df.drop('charges', axis=1)
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================
# MODELAGEM
# ==============================================
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results[name] = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    if hasattr(model, 'feature_importances_'):
        cat_features_encoded = pipeline.named_steps['preprocessor'].named_transformers_['cat']\
            .get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(cat_features_encoded)
        importances = pd.Series(model.feature_importances_, index=feature_names)

        plt.figure(figsize=(12, 8))
        importances.sort_values().tail(15).plot(kind='barh')
        plt.title(f'Feature Importance - {name}')
        plt.savefig(f'feature_importance_{name.lower().replace(" ", "_")}.png')
        plt.close()

results_df = pd.DataFrame(results).T
print("\n=== Comparação de Modelos ===")
print(results_df)

# ==============================================
# OTIMIZAÇÃO DO XGBOOST
# ==============================================
# param_grid = {
#     'regressor__n_estimators': [100, 150, 200],  # Menor número de estimadores
#     'regressor__learning_rate': [0.05, 0.1],  # Taxa de aprendizado
#     'regressor__max_depth': [2, 3],  # Reduzindo a profundidade das árvores
#     'regressor__min_child_weight': [1, 5],  # Aumentar valor pode ajudar a reduzir overfitting
#     'regressor__gamma': [0, 0.1, 0.2],  # Ajustando gamma para controle de divisão das árvores
#     'regressor__subsample': [0.7, 0.8, 1.0],  # Ajustando a amostragem
#     'regressor__colsample_bytree': [0.7, 0.8],  # Ajustando o número de features por árvore
#     'regressor__alpha': [0.1, 0.3],  # Regularização L1
#     'regressor__lambda': [0.1, 0.3]  # Regularização L2
# }

# param_grid = {
#     'regressor__n_estimators': [100, 200],
#     'regressor__learning_rate': [0.05, 0.1],
#     'regressor__max_depth': [2, 3],  # Reduzimos a profundidade máxima
#     'regressor__subsample': [0.8, 1.0],
#     # 'regressor__alpha': [0.1, 0.3],  # Regularização L1
#     # 'regressor__lambda': [0.1, 0.3]  # Regularização L2
#     'regressor__reg_alpha': [0.1, 0.3],  # Nome correto para L1
#     'regressor__reg_lambda': [0.1, 0.3]   # Nome correto para L2
# }

param_grid = {
    'regressor__n_estimators': [100, 150],
    'regressor__learning_rate': [0.01, 0.05],  # Reduza a taxa de aprendizado
    'regressor__max_depth': [2, 3],            # Profundidade menor
    'regressor__subsample': [0.7, 0.8],        # Menos amostras por árvore
    'regressor__reg_alpha': [0.5, 1],           # Regularização L1 mais forte
    'regressor__reg_lambda': [0.5, 1],          # Regularização L2 mais forte
    'regressor__gamma': [0.1, 0.3]              # Aumente para evitar splits desnecessários
}

best_model = GridSearchCV(
    Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42))
    ]),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=2
)

best_model.fit(X_train, y_train)

final_pred = best_model.best_estimator_.predict(X_test)

print("\n=== Melhor Modelo (XGBoost Otimizado) ===")
print(f"Melhores parâmetros: {best_model.best_params_}")
print(f"MAE: {mean_absolute_error(y_test, final_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, final_pred)):.2f}")
print(f"R²: {r2_score(y_test, final_pred):.4f}")

# cv_scores = cross_val_score(best_model.best_estimator_, X, y, cv=5, scoring='neg_mean_squared_error')

scoring = {'mae': 'neg_mean_absolute_error', 'mse': 'neg_mean_squared_error'}
cv_results = cross_validate(best_model, X, y, cv=5, scoring=scoring)

# Média do erro de validação cruzada
# print(f"\n=== Performance com Cross-Validation ===")
# print(f"CV MAE: {-cv_scores.mean():.2f}")
# print(f"CV RMSE: {np.sqrt(-cv_scores.mean()):.2f}")

print(f"CV MAE: {-cv_results['test_mae'].mean():.2f}")
print(f"CV RMSE: {np.sqrt(-cv_results['test_mse'].mean()):.2f}")

# ==============================================
# PDP - Partial Dependence
# ==============================================
# Usar colunas originais para o PDP
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    best_model.best_estimator_,
    X_train,
    features=['age', 'bmi', 'smoker'],
    ax=ax
)
plt.title('Partial Dependence Plots')
plt.savefig('partial_dependence.png')
plt.close()

drift_report = pd.DataFrame({
    'feature': X.columns,
    'data_drift': np.random.uniform(0, 0.2, len(X.columns))
})
