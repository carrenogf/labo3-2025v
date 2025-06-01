import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os

def optimize_memory_usage(df, verbose=True, convert_bool=True):
    """
    Optimiza el uso de memoria de un DataFrame reduciendo tipos numéricos, booleanos, categóricos y fechas.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame original a optimizar.
    verbose : bool, opcional (default=True)
        Si True, muestra información sobre la reducción de memoria.
    convert_bool : bool, opcional (default=True)
        Si True, convierte columnas binarias a booleanas.
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame optimizado con tipos de datos reducidos.
    """
    import pandas as pd
    import numpy as np

    df_optimized = df.copy()
    start_mem = df_optimized.memory_usage(deep=True).sum() / 1024**2

    for col in df_optimized.columns:
        col_type = df_optimized[col].dtypes

        if pd.api.types.is_numeric_dtype(col_type):
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()

            if pd.api.types.is_integer_dtype(col_type):
                if col_min >= 0:
                    if col_max <= np.iinfo(np.uint8).max:
                        df_optimized[col] = df_optimized[col].astype(np.uint8)
                    elif col_max <= np.iinfo(np.uint16).max:
                        df_optimized[col] = df_optimized[col].astype(np.uint16)
                    elif col_max <= np.iinfo(np.uint32).max:
                        df_optimized[col] = df_optimized[col].astype(np.uint32)
                    else:
                        df_optimized[col] = df_optimized[col].astype(np.uint64)
                else:
                    if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                    else:
                        df_optimized[col] = df_optimized[col].astype(np.int64)

            elif pd.api.types.is_float_dtype(col_type):
                if not df_optimized[col].isnull().any():
                    if col_min >= np.finfo(np.float16).min and col_max <= np.finfo(np.float16).max:
                        df_optimized[col] = df_optimized[col].astype(np.float16)
                    elif col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)
                    else:
                        df_optimized[col] = df_optimized[col].astype(np.float64)
                else:
                    # Con NaNs, evitamos float16
                    if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)

        elif pd.api.types.is_object_dtype(col_type):
            num_unique = df_optimized[col].nunique()
            num_total = len(df_optimized[col])
            if num_unique / num_total < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')

        elif pd.api.types.is_bool_dtype(col_type):
            df_optimized[col] = df_optimized[col].astype('bool')

        elif convert_bool and df_optimized[col].dropna().nunique() == 2:
            # Convertir columnas binarias a booleanas si no lo son aún
            unique_vals = df_optimized[col].dropna().unique()
            if set(unique_vals) <= {0, 1} or set(unique_vals) <= {True, False}:
                df_optimized[col] = df_optimized[col].astype('bool')

        elif pd.api.types.is_datetime64_any_dtype(col_type):
            # Ya está optimizada
            continue

        elif col_type == 'object':
            try:
                parsed_dates = pd.to_datetime(df_optimized[col], errors='coerce')
                if parsed_dates.notna().sum() > 0.9 * len(df_optimized[col]):
                    df_optimized[col] = parsed_dates
            except Exception:
                pass

    end_mem = df_optimized.memory_usage(deep=True).sum() / 1024**2

    if verbose:
        print(f"Memoria inicial: {start_mem:.2f} MB")
        print(f"Memoria final:   {end_mem:.2f} MB")
        print(f"Reducción:       {100 * (start_mem - end_mem) / start_mem:.2f}%")

    return df_optimized


os.makedirs("optuna_storage", exist_ok=True)
DB_PATH = "optuna_storage/optuna_simple.db"
STUDY_NAME = "17m_fe2"
storage_url = f"sqlite:///{DB_PATH}"


df = pd.read_csv('../datasets/dt_fe2.csv',sep=',')
# Optimizar el uso de memoria del DataFrame
df = optimize_memory_usage(df)

# === 1. Carga de datos y preprocesamiento ===
# Asegurate de tener cargado tu DataFrame `df`

df = df.drop(['periodo_dt', 'descripcion'], axis=1, errors='ignore')
df_kgl = df[df["periodo"] == 201912]
df = df[~df["periodo"].isin([201911, 201912])]

# Codificar categóricas
cat_cols = ['cat1', 'cat2', 'cat3', 'brand', 'plan_precios_cuidados']
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Separar features y target
X = df.drop(columns=["target"])
y = df["target"]

# Split fijo para validación (sin CV)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 2. Definición del objetivo para Optuna ===
def objective(trial):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "seed": 42,
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
    }

    model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_train),
        valid_sets=[lgb.Dataset(X_val, label=y_val)],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)  # MSE normal
    rmse = np.sqrt(mse)
    return rmse

# === 3. Configurar almacenamiento SQLite para Optuna ===


# === 4. Crear o cargar estudio ===
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=storage_url,
    direction="minimize",
    load_if_exists=True
)

# === 5. Ejecutar optimización ===
study.optimize(objective, n_trials=50)

# === 6. Mostrar resultados ===
print("Mejores hiperparámetros encontrados:")
print(study.best_params)
print(f"Mejor RMSE: {study.best_value:.4f}")