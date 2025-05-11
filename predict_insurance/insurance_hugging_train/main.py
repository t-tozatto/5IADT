import math

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Cols:
    GENDER = "Gender"
    SMOKING_STATUS = "Smoking Status"
    MARITAL_STATUS = "Marital Status"
    AGE = "Age"
    POLICY_TYPE = "Policy Type"
    POLICY_START_DATE = "Policy Start Date"
    INSURANCE_DURATION = "Insurance Duration"
    HEALTH_SCORE = "Health Score"
    OCCUPATION = "Occupation"
    EDUCATION_LEVEL = "Education Level"
    VEHICLE_AGE = "Vehicle Age"
    ANNUAL_INCOME = "Annual Income"
    CREDIT_SCORE = "Credit Score"
    PROPERTY_TYPE = "Property Type"
    NUMBER_OF_DEPENDENTS = "Number of Dependents"
    EXERCISE_FREQUENCY = "Exercise Frequency"
    PREVIOUS_CLAIMS = "Previous Claims"
    PREMIUM_AMOUNT = "Premium Amount"

SHOW_GRAPHICS = 0

#para mostrar todas as colunas no terminal
pd.set_option('display.max_columns', None)

## ETAPA 1 - explorando e entendendo os dados
# lendo o arquivo
df = pd.read_csv('train.csv', nrows=30000)

#print nas primeiras linhas do arquivo
print('\nHEAD')
print(df.head())

# recuperando os tipos das colunas, total de entradas, se tem null ou não
df.info()

#estatisticas das infos. para ajudar a perceber destribuição, outliers, assimetria..
#possivel utilizar o include='all' para incluir colunas categoricas
print('\nDESCRIBE')
print(df.describe())

# Selecionar apenas colunas categóricas
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns


# Iterar e imprimir informações
for col in categorical_cols:
    if col != Cols.POLICY_START_DATE:
        print(f'\n📌 Coluna: {col}')
        print(f'Valores únicos: {set(df[col])}')
        print('Contagem de valores:')
        print(df[col].value_counts())

#kde true para mostrar a curva de densidade
#histplot
if SHOW_GRAPHICS == 1:
    # Distribuição das variáveis numéricas
    num_cols = df.select_dtypes(include='number').columns
    total = len(num_cols)

    cols = 3
    rows = math.ceil(total / cols)

    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(data=df, x=col, kde=True, bins=30, color='skyblue')
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')

    plt.tight_layout()
    plt.savefig('distribuicao_numbericos.png')
    plt.close()

    # Distribuição das variáveis categóricas
    cat_cols = df.select_dtypes(include='object').columns

    total = len(cat_cols)
    rows = math.ceil(total / cols)
    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, col in enumerate(cat_cols, 1):
        plt.subplot(rows, cols, i)  # Ajustado para usar rows e cols
        sns.countplot(x=col, data=df, legend=False)
        plt.title(f'Contagem de {col}')
        plt.xticks(rotation=45)
        plt.xlabel(col)
        plt.ylabel('Frequência')

    plt.tight_layout()
    plt.savefig('distribuicao_categoricos.png')
    plt.close()

    # Gráfico de distribuição percentual para variáveis categóricas
    total = len(cat_cols)
    rows = math.ceil(total / cols)

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, col in enumerate(cat_cols, 1):
        plt.subplot(rows, cols, i)  # Ajustado para usar rows e cols
        sns.countplot(x=col, data=df)

        col_total = df[col].value_counts().sum()  # Ajustado para o total correto
        for p in plt.gca().patches:
            height = p.get_height()
            plt.gca().text(p.get_x() + p.get_width() / 2., height + 10, f'{height / col_total:.2%}', ha="center")

        plt.title(f'Distribuição de {col}')
        plt.xticks(rotation=45)
        plt.xlabel(col)
        plt.ylabel('Frequência')

    plt.tight_layout()
    plt.savefig('distribuicao_percentual_categoricos.png')
    plt.close()

    # Mapa de dados faltantes
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Mapa de Dados Faltantes')
    plt.savefig('mapa_dados_faltantes.png')
    plt.close()

    # Boxplot de variáveis categóricas vs. Premium Amount
    total = len(cat_cols)
    rows = math.ceil(total / cols)
    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, col in enumerate(cat_cols, 1):
        plt.subplot(rows, cols, i)  # Ajustado para usar rows e cols
        sns.boxplot(data=df, x=col, y=Cols.PREMIUM_AMOUNT)
        plt.title(f'{col} vs Premium Amount')

    plt.tight_layout()
    plt.savefig('boxplot_categoricas_vs_amount.png')
    plt.close()

    # Verificando a distribuição das apólices ao longo do tempo
    df[Cols.POLICY_START_DATE] = pd.to_datetime(df[Cols.POLICY_START_DATE], errors='coerce')
    df['Year'] = df[Cols.POLICY_START_DATE].dt.year

    plt.figure(figsize=(12, 6))
    sns.countplot(x='Year', data=df)
    plt.title('Distribuição de Apólices ao Longo do Tempo')
    plt.xlabel('Ano')
    plt.ylabel('Contagem de Apólices')
    plt.savefig('distribuicao_apolices_tempo.png')
    plt.close()

## ETAPA 2 - pre-processamento

#count nos nulls
print('\nCount nulls')
print(df.isnull().sum())

# Tratamento de datas
df[Cols.POLICY_START_DATE] = pd.to_datetime(df[Cols.POLICY_START_DATE], errors='coerce')
df[Cols.POLICY_START_DATE].fillna(df[Cols.POLICY_START_DATE].mean(), inplace=True)

df_ref = df[Cols.POLICY_START_DATE].max()
df['policy_usage_in_days'] = (df_ref - df[Cols.POLICY_START_DATE]).dt.days
df['policy_usage_in_years'] = df['policy_usage_in_days'] / 365.25
df.drop(columns=[Cols.POLICY_START_DATE, 'policy_usage_in_days'], inplace=True)

#tratando variáveis de 2 valores
# Inicializando o LabelEncoder
le = LabelEncoder()

# Identificar colunas categóricas com 2 valores únicos
cat_cols = df.select_dtypes(include='object').columns
binary_cols = [col for col in cat_cols if len(df[col].unique()) == 2]

# Aplicar LabelEncoder nas colunas binárias
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Identificar colunas categóricas com mais de 2 valores únicos (nominais)
nominal_cols = [col for col in cat_cols if len(df[col].unique()) > 2]

# Usando o OneHotEncoder para variáveis nominais
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# Identificar as colunas numéricas
numerical_cols = df.select_dtypes(include='number').columns

# Inicializar o scaler (escolhendo entre StandardScaler ou MinMaxScaler)
scaler = StandardScaler()  # Ou use MinMaxScaler() se preferir

# Aplicar o scaler nas variáveis numéricas
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Verificando as primeiras linhas após a transformação
print(df.head())

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['policy_usage_in_years'])
plt.title('Boxplot de Uso da Apólice em Anos')
plt.savefig('boxplot_apolices_tempo.png')
plt.close()

# annot=True mostra os valores
# cmap=coolwarm cores para identificar forças de correlação
# azul negativo, vermelho positivo
# 1=correlacao perfeita, 0=nenhuma correlacao, -1=correlação negativa perfeita
corr = df.corr(numeric_only=True)

# Define o tamanho da figura com base no número de variáveis
plt.figure(figsize=(14, 12))  # ajuste conforme necessário

# Cria o mapa de calor
sns.heatmap(
    corr,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5,
    cbar=True,
    annot_kws={"size": 8}  # reduz o tamanho da fonte dos números
)

plt.title('Matriz de Correlação', fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

plt.savefig('matriz_correlacao.png')
plt.close()

# converte bool para int
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print('\nHEAD')
print(df.head())

## ETAPA 3 - modelagem

#removendo coluna alvo, x=preditoras
X = df.drop(Cols.PREMIUM_AMOUNT, axis=1)

#separando a coluna alvo, y=alvo
y = df[Cols.PREMIUM_AMOUNT]

#test_size = porcentagem do teste (20%)
#random_state - seed do gerador de números aleatórios, controla aleatoriedade
#42 = a vida, o universo e tudo mais - poderia ser qualquer número inteiro
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# # Normalização
# # Identificar colunas numéricas para normalizar
# numeric_cols = ['age', 'bmi', 'children']  # ajuste conforme seu dataset

# # Aplicar o scaler apenas nelas
# scaler = StandardScaler()
# X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
# X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

#RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

#fit o modelo aos dados de entrada e aprende os padrões que relacionam suas variáveis (X) com o que você quer prever (y)
#analisar os dados de X_train
#aprender como eles influenciam y_train (os custos médicos)
#calcular os coeficientes da equação da reta (porque regressão linear = equação do tipo y = aX + b)
model.fit(X_train, y_train)

#faz as previsões
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print('\nMétricas')

#MAE — Mean Absolute Error (Erro Absoluto Médio)
print(f"MAE:  {mae:.2f}")

#RMSE — Root Mean Squared Error (Raiz do Erro Quadrático Médio)
#se RMSE for muito maior que o MAE, pode significar que seu modelo está errando feio em alguns casos específicos (outliers).
print(f"RMSE: {rmse:.2f}")

#Coeficiente de Determinação
# 0 a 1 (às vezes pode até ser negativo se o modelo for muito ruim)
#Quanto mais próximo de 1, melhor o modelo está.
#Se estiver muito baixo (tipo < 0.5), o modelo pode estar errando bastante ou não está captando bem os padrões.
print(f"R²:   {r2:.4f}")
