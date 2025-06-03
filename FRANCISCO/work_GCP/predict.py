import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
from reducirDT import optimize_memory_usage
import datetime
import random
import os
import gc

DATASET_NAME = "dt_fe"

DATASET_PATH = f"../datasets/{DATASET_NAME}.csv"
STUDY_NAME = "lightgbm_sin_cv"

today = datetime.datetime.now()
today_str = today.strftime("%Y-%m-%d")

RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

result_filename = f"resultados-{today_str}.csv"
while os.path.exists(os.path.join(RESULTS_DIR, result_filename)):
    nrandom = random.randint(1, 100000)
    result_filename = f"resultados-{today_str}_{nrandom}.csv"

result_path = os.path.join(RESULTS_DIR, result_filename)
    


df = pd.read_csv(DATASET_PATH, sep=',')

# Optimizar el uso de memoria del DataFrame
df = optimize_memory_usage(df)

df2 = pd.read_csv('../datasets/dt_fe2.csv',sep=',')

df2 = optimize_memory_usage(df2)
df["proporcion_producto_en_total_mes"] = df2["proporcion_producto_en_total_mes"]
df["modo_diff"] = df2["modo_diff"]
df["modo_6m"] = df2["modo_6m"]
df["std_6m"] = df2["std_6m"]

del df2
gc.collect()


# Codificar categóricas
print("Codificando categóricas...")
cat_cols = ['cat1', 'cat2', 'cat3', 'brand', 'plan_precios_cuidados']
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

df = df.drop(['periodo_dt'], axis=1, errors='ignore')

print("Separando datos de entrenamiento y validación...")

df_kgl = df[df["periodo"] == 201912]
df = df[~df["periodo"].isin([201911, 201912])]
# Separar features y target
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)


# Ruta a la base de datos
DB_PATH = "../notebooks/optuna_storage/optuna_simple.db"

# Cargar el estudio almacenado
study = optuna.load_study(
    study_name="lightgbm_sin_cv",  # Usar None si solo hay un estudio en la DB, o especificar el nombre
    storage=f"sqlite:///{DB_PATH}"
)

# Obtener los mejores hiperparámetros
best_params = study.best_params
best_value = study.best_value

print("Mejores hiperparámetros encontrados:")
print(best_params)
print(f"Mejor valor objetivo: {best_value}")


# Entrenar con los mejores hiperparámetros
best_params = study.best_params.copy()
best_params.update({
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42
})

print("Entrenando el modelo con los mejores hiperparámetros...")
# Entrenar una vez el modelo con esos parámetros (ej. sobre X_train si no querés usar todo)
model = lgb.train(
    best_params,
#     {
#     "objective": "regression",
#     "metric": "rmse",
#     "verbosity": -1,
#     "n_jobs": -1,
#     "seed": 42
# },
    lgb.Dataset(X_train, label=y_train),
    valid_sets=[lgb.Dataset(X_val, label=y_val)],
    num_boost_round=1000,
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)

print("Modelo entrenado.")
print("Haciendo predicciones sobre el set de validación...")
# Asegurar las mismas columnas
X_kgl = df_kgl[X.columns]  # Misma estructura

# === 9. Hacer predicción sobre nuevos datos ===
preds_kgl = model.predict(X_kgl)


productos_ok = pd.read_csv("https://storage.googleapis.com/open-courses/austral2025-af91/labo3v/product_id_apredecir201912.txt", sep="\t")
result_df = pd.DataFrame({"product_id": X_kgl["product_id"], "tn": X_kgl["tn"],  "ypred": preds_kgl})
result_df["tn"] = result_df["ypred"] + result_df["tn"]
result_df = result_df[result_df["product_id"].isin(productos_ok["product_id"])]
result_df = result_df.groupby("product_id").agg({"tn":"sum"}).reset_index()
print("Predicciones realizadas.")
print("Guardando resultados...")
result_df.to_csv(result_path, index=False,sep=",")

bucket = "gs://resultados_labo3/"

ret = os.system(f'gsutil cp {result_path} {bucket}')
if ret != 0:
    print("❌ Error al subir el archivo al bucket de GCS")
else:
    print("✅ Archivo subido correctamente al bucket de GCS")