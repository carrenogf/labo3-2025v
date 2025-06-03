import pandas as pd
from reducirDT import optimize_memory_usage
from scipy.stats import mode

dt_fe = "dt_fe3"

df = pd.read_csv('../datasets/dt_target.csv',sep=',')
df = optimize_memory_usage(df)
df["target"] = df["target"].fillna(0)

# Ordenar por producto, cliente y tiempo
df["periodo_dt"] = pd.to_datetime(df["periodo_dt"], format='%Y-%m-%d')
df = df.sort_values(['product_id', 'customer_id', 'periodo_dt'])


df["month"] = df["periodo_dt"].dt.month
df["year"] = df["periodo_dt"].dt.year
df['quarter'] = df['month'].apply(lambda x: (x-1)//3 + 1)
df['semester'] = df['month'].apply(lambda x: 1 if x <=6 else 2)
df['is_month_end'] = df['month'].isin([1, 3, 5, 7, 8, 10, 12])  # Meses con 31 días
df['season'] = df['month']%12 // 3 + 1  # 1:Invierno, 2:Primavera, etc.
df['size_vs_category'] = df['sku_size'] / df.groupby('cat3')['sku_size'].transform('mean')


# Crear lags
df['lag_1m'] = df.groupby(['product_id', 'customer_id'])['tn'].shift(1)
df['lag_2m'] = df.groupby(['product_id', 'customer_id'])['tn'].shift(2)
df['lag_3m'] = df.groupby(['product_id', 'customer_id'])['tn'].shift(3)
df['lag_11m'] = df.groupby(['product_id', 'customer_id'])['tn'].shift(11)

# Promedio móvil
df['rolling_3m_mean'] = df.groupby(['product_id', 'customer_id'])['tn'].transform(
    lambda x: x.rolling(3, min_periods=1).mean())

df['rolling_6m_mean'] = df.groupby(['product_id', 'customer_id'])['tn'].transform(
    lambda x: x.rolling(6, min_periods=1).mean())

df['rolling_12m_mean'] = df.groupby(['product_id', 'customer_id'])['tn'].transform(
    lambda x: x.rolling(12, min_periods=1).mean())


# Tendencia anual
df['annual_trend'] = df.groupby(['product_id', 'month'])['tn'].transform('mean')

# Variación estacional
df['seasonal_variation'] = df['tn'] / df['annual_trend']

# segunda tanda de features

# Desviación estándar y coeficiente de variación
df['std_6m'] = df.groupby(['product_id', 'customer_id'])['tn'].transform(
    lambda x: x.shift(1).rolling(6, min_periods=2).std())

# df['cv_6m'] = df['std_6m'] / (df['rolling_6m_mean'] + 1e-5)  # Coef. de variación: std/mean


# # Periodos desde última compra
# df['comprado'] = (df['tn'] > 0).astype(int)
# # df['periodos_desde_ultima_compra'] = df.groupby(['product_id', 'customer_id'])['comprado'].apply(
# #     lambda x: x[::-1].cumsum()[::-1].where(x==1).ffill().fillna(0))
# df['periodos_desde_ultima_compra'] = df.groupby(['product_id', 'customer_id'])['comprado'].transform(
#     lambda x: x[::-1].cumsum()[::-1].where(x==1).ffill().fillna(0))
# # Cantidad de meses con compra en últimos N meses
# for window in [3, 6, 12]:
#     df[f'freq_compra_{window}m'] = df.groupby(['product_id', 'customer_id'])['comprado'].transform(
#         lambda x: x.shift(1).rolling(window, min_periods=1).sum())
    


# # Modo de cantidad (más común en últimos 6 meses)
def rolling_mode(x):
    return x.shift(1).rolling(6, min_periods=1).apply(lambda x: mode(x, keepdims=True)[0][0], raw=False)
#Moda o patrón de cantidad
df['modo_6m'] = df.groupby(['product_id', 'customer_id'])['tn'].transform(rolling_mode)
df['modo_diff'] = df['tn'] - df['modo_6m']
# #Tendencia reciente
# df['trend_3m'] = df.groupby(['product_id', 'customer_id'])['tn'].transform(
#     lambda x: x.shift(1).rolling(3).apply(lambda y: y.iloc[-1] - y.iloc[0] if len(y) == 3 else 0))
# #Promedio histórico total
# df['media_historica_cliente_producto'] = df.groupby(['product_id', 'customer_id'])['tn'].transform(
#     lambda x: x.expanding().mean())
# #Ratio de compra cliente vs. total producto
# df['participacion_producto'] = df['tn'] / (df['periodo_producto'] + 1e-5)
# #Uso de stock
# df['stock_vs_venta'] = df['stock_final'] / (df['tn'] + 1e-5)

# df['meses_desde_nacimiento'] = df['periodo'] - df['nacimiento_producto']
# df['meses_hasta_muerte_cliente'] = df['muerte_cliente'] - df['periodo']
# df['productos_distintos_cliente_mes'] = df.groupby(['customer_id', 'periodo'])['product_id'].transform('nunique')
df['total_cliente_mes'] = df.groupby(['customer_id', 'periodo'])['tn'].transform('sum')
df['proporcion_producto_en_total_mes'] = df['tn'] / (df['total_cliente_mes'] + 1e-5)
# # tn vs. cust_request_tn → indicador de si le entregaron lo que pidió
# df['ratio_entregado_sobre_pedido'] = df['tn'] / (df['cust_request_tn'] + 1e-5)
# # tn vs. sku_size → toneladas por unidad
# df['tn_por_unidad'] = df['tn'] / (df['sku_size'] + 1e-5)

df.to_csv(f"../datasets/{dt_fe}.csv", index=False, sep=",")