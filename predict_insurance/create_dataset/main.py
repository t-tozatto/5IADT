import pandas as pd
import numpy as np
from scipy.stats import skewnorm

# ==============================================
# CONFIGURAÇÕES INICIAIS
# ==============================================
np.random.seed(42)
N_SAMPLES = 40_000
MISSING_PERCENT = 0.05

# ==============================================
# GERAR VARIÁVEIS BASE
# ==============================================

def generate_base_data(n_samples):
    income = np.clip(skewnorm.rvs(-4, loc=5000, scale=2500, size=n_samples), 2000, 50000)

    data = {
        'age': np.clip(np.random.normal(40, 12, n_samples), 18, 65).astype(int),
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.49, 0.51]),
        'region': np.random.choice(['north', 'south', 'east', 'west'], n_samples, p=[0.25, 0.35, 0.25, 0.15]),
        'bmi': np.clip(skewnorm.rvs(5, loc=25, scale=5, size=n_samples), 16, 40),
        'smoker': np.random.binomial(1, 0.2, n_samples),
        'chronic_conditions': np.random.poisson(0.7, n_samples),
        'genetic_risk': np.random.beta(2, 5, n_samples) * 100,
        'exercise_frequency': np.random.choice(
            ['sedentary', 'light', 'moderate', 'active'],
            n_samples,
            p=[0.3, 0.4, 0.2, 0.1]
        ),
        'alcohol_consumption': np.clip(np.random.poisson(2, n_samples), 0, 20),
        'diet_quality': np.where(income > 10000,
                                 np.random.choice([3, 4, 5], size=n_samples, p=[0.2, 0.5, 0.3]),
                                 np.random.choice([1, 2, 3], size=n_samples, p=[0.3, 0.4, 0.3])),
        'income': income,
        'occupation_risk': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'coverage_level': np.random.choice(['basic', 'standard', 'premium'], n_samples, p=[0.6, 0.25, 0.15]),
        'num_dependents': np.random.poisson(1.2, n_samples)
    }
    return pd.DataFrame(data)

# ==============================================
# ENGENHARIA DE FEATURES
# ==============================================

def add_engineered_features(df):
    # 3. Ajuste do BMI com base na frequência de exercício
    bmi_adjustment = {
        'sedentary': 0.5,
        'light': -0.5,
        'moderate': -1.5,
        'active': -2.5
    }
    df['bmi'] += df['exercise_frequency'].map(bmi_adjustment)
    df['bmi'] = df['bmi'].clip(16, 40)

    # 4. Reatribuição de cobertura para idosos
    older_mask = df['age'] > 55
    df.loc[older_mask, 'coverage_level'] = np.random.choice(
        ['standard', 'premium'],
        size=older_mask.sum(),
        p=[0.4, 0.6]
    )

    # Fatores regionais e de cobertura
    regional_cost = {'north': 1.15, 'south': 0.95, 'east': 1.05, 'west': 1.0}
    coverage_map = {'basic': 1.0, 'standard': 1.2, 'premium': 1.5}

    df['regional_cost_factor'] = df['region'].map(regional_cost)
    df['coverage_factor'] = df['coverage_level'].map(coverage_map)

    return df

# ==============================================
# GERAR TARGET (CHARGES)
# ==============================================

def generate_target(df):
    base = 200
    charges = (
        base
        + 80 * df['age'] ** 1.5
        + 3000 * df['smoker']
        + 1200 * df['chronic_conditions'] ** 2
        + 15 * np.abs(df['bmi'] - 25) ** 3
        - 200 * df['exercise_frequency'].map({
            'sedentary': 0,
            'light': 1,
            'moderate': 2,
            'active': 3
        })
        + 50 * df['alcohol_consumption'] ** 1.7
        + 0.05 * df['genetic_risk'] * df['age']
        + 0.0002 * df['income'] * df['regional_cost_factor']
        + 300 * df['occupation_risk']
        + 500 * df['num_dependents']
    )

    charges *= df['coverage_factor']

    noise = np.random.normal(0, 500 + df['age'] * 10)
    charges += noise

    return np.clip(charges, 500, 50000).astype(int)

# ==============================================
# VALORES AUSENTES
# ==============================================

def add_missing_values(df, missing_percent):
    cols_with_missing = ['genetic_risk', 'alcohol_consumption', 'diet_quality']
    for col in cols_with_missing:
        mask = np.random.rand(len(df)) < missing_percent
        df.loc[mask, col] = np.nan
    return df

# ==============================================
# NORMALIZAÇÃO DE NOMES (opcional para ML)
# ==============================================

def normalize_column_names(df):
    return df.rename(columns={
        'exercise_frequency': 'exercise',
        'diet_quality': 'diet',
        'num_dependents': 'dependents',
    })

# ==============================================
# PIPELINE PRINCIPAL
# ==============================================

if __name__ == "__main__":
    df = generate_base_data(N_SAMPLES)
    df = add_engineered_features(df)
    df['charges'] = generate_target(df)
    df = add_missing_values(df, MISSING_PERCENT)

    df_model = normalize_column_names(df)

    df_model.to_csv('health_insurance_dataset.csv', index=False)
    print("✅ Dataset gerado com sucesso! Formato:", df_model.shape)
    print("\n🔍 Amostra do dataset:")
    print(df_model.sample(3))
