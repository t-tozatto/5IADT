import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SHOW_GRAPHICS = 0

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

#kde true para mostrar a curva de densidade
#histplot
if SHOW_GRAPHICS == 1:
    sns.displot(df['charges'], kde=True)
    plt.title("Distribuição dos custos médicos")
    plt.xlabel("Encargos")
    plt.ylabel("Frequência")
    plt.show()

# annot=True mostra os valores
# cmap=coolwarm cores para identificar forças de correlação
# azul negativo, vermelho positivo
# 1=correlacao perfeita, 0=nenhuma correlacao, -1=correlação negativa perfeita
if SHOW_GRAPHICS == 1:
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Matriz de Correlação")
    plt.show()

## ETAPA 2 - pre-processamento

#count nos nulls
print('\nCount nulls')
print(df.isnull().sum())

#pega variáveis com mais de 2 valores (não é sim/não..)
#drop_first=True evita duplicidade de info
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# converte bool para int
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

#manualmente fazendo a map das variáveis de 2 valores
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

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


