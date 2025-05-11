# insurance_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===========================================
# CONFIGURAÇÕES
# ===========================================
class Config:
    INPUT_PATH = "train.csv"
    OUTPUT_DIR = "results"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    PLOT_STYLE = "seaborn-v0_8"
    FIG_SIZE = (12, 6)


# ===========================================
# FUNÇÕES AUXILIARES
# ===========================================
def setup_environment():
    """Configura ambiente e diretórios"""
    plt.style.use(Config.PLOT_STYLE)
    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
    pd.set_option('display.max_columns', 50)


def save_fig(filename):
    """Salva gráficos em alta resolução"""
    plt.savefig(
        f"{Config.OUTPUT_DIR}/{filename}",
        bbox_inches='tight',
        dpi=300,
        facecolor='white'
    )
    plt.close()


# ===========================================
# ANÁLISE E LIMPEZA DOS DADOS
# ===========================================
class DataCleaner:
    @staticmethod
    def load_and_clean():
        try:
            df = pd.read_csv(Config.INPUT_PATH)
        except FileNotFoundError:
            raise SystemExit("❌ Arquivo não encontrado. Verifique o caminho do dataset.")

        # Limpeza inicial
        df = df.drop(columns=['id']) if 'id' in df.columns else df

        # Converter datas
        if 'Policy Start Date' in df.columns:
            df['Policy Start Date'] = pd.to_datetime(
                df['Policy Start Date'],
                errors='coerce'
            )
            ref_date = df['Policy Start Date'].max() + pd.DateOffset(days=1)
            df['Policy Age (days)'] = (ref_date - df['Policy Start Date']).dt.days
            df = df.drop(columns=['Policy Start Date'])

        # Tratar missing values
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(exclude=np.number).columns

        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[cat_cols] = df[cat_cols].fillna('missing')

        return df


# ===========================================
# VISUALIZAÇÃO DOS DADOS
# ===========================================
class DataVisualizer:
    @staticmethod
    def plot_distributions(df):
        """Gráficos de distribuição"""
        # Distribuição do target
        plt.figure(figsize=Config.FIG_SIZE)
        sns.histplot(df['Premium Amount'], kde=True, bins=30)
        plt.title('Distribuição do Valor do Prêmio')
        save_fig('distribuicao_premio.png')

        # Relação Idade vs Prêmio
        plt.figure(figsize=Config.FIG_SIZE)
        sns.scatterplot(
            data=df,
            x='Age',
            y='Premium Amount',
            hue='Smoking Status',
            alpha=0.6,
            palette='viridis'
        )
        plt.title('Relação Idade vs Valor do Prêmio')
        save_fig('idade_vs_premio.png')

        # Correlações
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            df.select_dtypes(include=np.number).corr(numeric_only=True),
            annot=True,
            fmt='.2f',
            cmap='coolwarm'
        )
        plt.title('Matriz de Correlação')
        save_fig('correlacoes.png')

    @staticmethod
    def plot_categorical_analysis(df):
        """Análise de variáveis categóricas"""
        cat_cols = df.select_dtypes(exclude=np.number).columns

        for col in cat_cols:
            plt.figure(figsize=Config.FIG_SIZE)
            sns.boxplot(
                data=df,
                x=col,
                y='Premium Amount',
                palette='Set3'
            )
            plt.xticks(rotation=45)
            plt.title(f'Relação {col} vs Prêmio')
            save_fig(f'boxplot_{col}.png')


# ===========================================
# MODELAGEM
# ===========================================
class InsuranceModel:
    def __init__(self):
        self.numeric_features = [
            'Age',
            'Annual Income',
            'Health Score',
            'Policy Age (days)'
        ]

        self.categorical_features = [
            'Gender',
            'Marital Status',
            'Smoking Status',
            'Policy Type'
        ]

        self.preprocessor = self._build_preprocessor()
        self.model = self._build_model()

    def _build_preprocessor(self):
        """Constroi pipeline de pré-processamento"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        return ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])

    def _build_model(self):
        """Constroi pipeline completo"""
        return Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=Config.RANDOM_STATE
            ))
        ])

    def evaluate(self, X_test, y_test):
        """Avaliação do modelo"""
        y_pred = self.model.predict(X_test)

        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R²': r2_score(y_test, y_pred)
        }

        # Gráfico de resíduos
        plt.figure(figsize=Config.FIG_SIZE)
        sns.residplot(
            x=y_pred,
            y=y_test - y_pred,
            lowess=True,
            line_kws={'color': 'red'}
        )
        plt.title('Análise de Resíduos')
        plt.xlabel('Valores Preditos')
        plt.ylabel('Resíduos')
        save_fig('residuos.png')

        return metrics


# ===========================================
# EXECUÇÃO PRINCIPAL
# ===========================================
if __name__ == "__main__":
    setup_environment()

    # Carregar e limpar dados
    print("🔎 Carregando e limpando dados...")
    df = DataCleaner.load_and_clean()

    # Análise exploratória
    print("📊 Gerando visualizações...")
    DataVisualizer.plot_distributions(df)
    DataVisualizer.plot_categorical_analysis(df)

    # Modelagem
    print("🤖 Construindo modelo...")
    model = InsuranceModel()

    # Preparar dados
    X = df.drop(columns=['Premium Amount'])
    y = df['Premium Amount']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )

    # Treinar
    print("🏋️ Treinando modelo...")
    model.model.fit(X_train, y_train)

    # Avaliar
    print("🧪 Avaliando performance...")
    metrics = model.evaluate(X_test, y_test)

    print("\n📈 Métricas Finais:")
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}")

    print(f"\n✅ Análise concluída! Gráficos salvos em: {Config.OUTPUT_DIR}/")