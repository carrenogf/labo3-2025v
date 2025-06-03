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
