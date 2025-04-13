import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

SHOW_GRAPHICS = 1

#para mostrar todas as colunas no terminal
pd.set_option('display.max_columns', None)

## ETAPA 1 - explorando e entendendo os dados
# lendo o arquivo
df = pd.read_csv('insurance.csv')

#print nas primeiras linhas do arquivo
print('\nHEAD')
print(df.head())

# recuperando os tipos das colunas, total de entradas, se tem null ou não
df.info()

#estatisticas das infos. para ajudar a perceber destribuição, outliers, assimetria..
#possivel utilizar o include='all' para incluir colunas categoricas
print('\nDESCRIBE')
print(df.describe())

print('\nSex')
print(set(df['sex']))
print(df['sex'].value_counts())

print('\nSmoker')
print(set(df['smoker']))
print(df['smoker'].value_counts())

print('\nRegion')
print(set(df['region']))
print(df['region'].value_counts())

#kde true para mostrar a curva de densidade
#histplot
if SHOW_GRAPHICS == 1:
    # Seleciona colunas numéricas automaticamente
    num_cols = df.select_dtypes(include='number').columns

    # Define o tamanho da figura e o grid de subplots
    plt.figure(figsize=(15, 10))

    # Para cada coluna numérica, cria um histograma
    for i, col in enumerate(num_cols, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data=df, x=col, kde=True, bins=30, color='skyblue')
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')

    plt.tight_layout()
    plt.show()

    # Seleciona as colunas categóricas automaticamente
    cat_cols = df.select_dtypes(include='object').columns

    # Define o tamanho da figura e o grid
    plt.figure(figsize=(15, 8))

    # Cria um gráfico para cada coluna categórica
    for i, col in enumerate(cat_cols, 1):
        plt.subplot(1, len(cat_cols), i)  # Tudo em uma linha
        sns.countplot(x=col, data=df, palette='pastel', hue=col, legend=False)
        plt.title(f'Contagem de {col}')
        plt.xticks(rotation=45)
        plt.xlabel(col)
        plt.ylabel('Frequência')

    plt.tight_layout()
    plt.show()

    # Calcular médias de charges por região
    print(df.groupby('region')['charges'].mean().sort_values(ascending=False))

## ETAPA 2 - pre-processamento

#count nos nulls
print('\nCount nulls')
print(df.isnull().sum())

#manualmente fazendo a map das variáveis de 2 valores
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# annot=True mostra os valores
# cmap=coolwarm cores para identificar forças de correlação
# azul negativo, vermelho positivo
# 1=correlacao perfeita, 0=nenhuma correlacao, -1=correlação negativa perfeita
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()

#pega variáveis com mais de 2 valores (não é sim/não..)
df = pd.get_dummies(df, columns=['region'], drop_first=False)

# converte bool para int
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

print('\nHEAD')
print(df.head())

## ETAPA 3 - modelagem

#removendo coluna alvo, x=preditoras
X = df.drop('charges', axis=1)

#separando a coluna alvo, y=alvo
y = df['charges']

#test_size = porcentagem do teste (20%)
#random_state - seed do gerador de números aleatórios, controla aleatoriedade
#42 = a vida, o universo e tudo mais - poderia ser qualquer número inteiro
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalização
# Identificar colunas numéricas para normalizar
numeric_cols = ['age', 'bmi', 'children']  # ajuste conforme seu dataset

# Aplicar o scaler apenas nelas
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

#regressão linear
model = LinearRegression()

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
