import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.transform import Rotation

 
# Estimación de clusters por la desviación estándar
def estimar_k_por_std(X, max_clusters=5, threshold_std=0.01):
    std_error = np.std(X)
    if std_error < threshold_std:
        return 1  # Si el error es bajo, usamos un solo cluster
    else:
        return min(max_clusters, max(2, int(std_error * 1000)))  # Calculamos el número de clusters basado en el error


def etiquetar_clusters_fijo(df, col_cluster="cluster"):
    etiquetas = {}
    
    # Asignar etiquetas fijas según el número de cluster
    etiquetas_fijas = {
        0: "muy bueno",
        1: "bueno",
        2: "regular",
        3: "malo",
        4: "muy malo"
    }

    # Asignar etiqueta a cada cluster basado en el valor
    for cluster in df[col_cluster].unique():
        if cluster in etiquetas_fijas:
            etiquetas[cluster] = etiquetas_fijas[cluster]
        else:
            etiquetas[cluster] = "desconocido"  # Para clusters que no estén en el rango esperado

    return etiquetas


# Función principal de clustering
def agregar_clusters_por_error(dic_diferencias, max_clusters=5, random_state=42, std_threshold=0.01):
    dic_clusterizado = {}

    for articulacion, df in dic_diferencias.items():
        df_clusterizado = df.copy()

        # Calcular el error si no está
        if "error" not in df_clusterizado.columns:
            df_clusterizado["error"] = np.sqrt(df_clusterizado["dx"]**2 +
                                               df_clusterizado["dy"]**2 +
                                               df_clusterizado["dz"]**2)

        n_nan = df_clusterizado["error"].isna().sum()
        if n_nan > 0:
            print(f"[AVISO] {articulacion}: {n_nan} NaNs encontrados. Se eliminarán filas.")
            df_clusterizado = df_clusterizado.dropna(subset=["error"])

        X = df_clusterizado[["error"]].values

        if len(X) < 3:
            print(f"[ADVERTENCIA] {articulacion}: muy pocos datos para clusterizar. Se omitirá.")
            continue

        n_clusters = estimar_k_por_std(X, max_clusters)

        # Realizar clustering
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)

        df_clusterizado["cluster"] = labels

        # Ordenar los errores y asignar el cluster con el menor error como el cluster 0
        error_promedio = df_clusterizado.groupby("cluster")["error"].mean().sort_values()
        cluster_a_ordenar = error_promedio.index.tolist()

        # Reordenar los clusters según el error promedio, asignando el menor error a 0, el siguiente a 1, etc.
        cluster_mapeo = {cluster_a_ordenar[i]: i for i in range(len(cluster_a_ordenar))}
        df_clusterizado["cluster"] = df_clusterizado["cluster"].map(cluster_mapeo)

        # Asignar etiquetas fijas según el número de cluster
        etiquetas = etiquetar_clusters_fijo(df_clusterizado, col_cluster="cluster")
        df_clusterizado["etiqueta"] = df_clusterizado["cluster"].map(etiquetas)

        dic_clusterizado[articulacion] = df_clusterizado

    return dic_clusterizado


def agrupar_por_ventanas_ms(dic_clusterizado, duracion_ventana_ms=10):
    """
    Agrupa los datos de cada articulación en ventanas de duración específica en milisegundos.
    
    :param dic_clusterizado: Diccionario con los datos clusterizados para cada articulación.
    :param duracion_ventana_ms: Duración de cada ventana en milisegundos.
    :return: Diccionario con los resultados agrupados por ventana.
    """
    resultados_ventanas = {}
    resultados_ventana_posicion = {}
    etiquetas_interes = ['muy malo', 'malo', 'regular', 'bueno', 'muy bueno']
    print("Resultados posicion por ventana:")
    for articulacion, df in dic_clusterizado.items():
        resultados_ventana_posicion[articulacion] = []
        print(f"\n📌 Articulación: {articulacion}")

        df = df.sort_values("ms").reset_index(drop=True)
        resultados_ventanas[articulacion] = []

        ms_inicio_total = df["ms"].min()
        ms_final_total = df["ms"].max()

        ms_actual = ms_inicio_total
        while ms_actual < ms_final_total:
            ms_limite = ms_actual + duracion_ventana_ms
            ventana = df[(df["ms"] >= ms_actual) & (df["ms"] < ms_limite)]

            if len(ventana) == 0:
                ms_actual = ms_limite
                continue

            ms_inicio = int(ventana["ms"].iloc[0])
            ms_final = int(ventana["ms"].iloc[-1])
            frames_inicial = int(ventana["frames"].iloc[0])
            frames_final = int(ventana["frames"].iloc[-1])
            duracion_ms = ms_final - ms_inicio

            etiqueta_mas_comun = Counter(ventana["etiqueta"]).most_common(1)[0][0]
            error_promedio = ventana["error"].mean()

            resultados_ventanas[articulacion].append({
                "ms_inicio": ms_inicio,
                "ms_final": ms_final,
                "etiqueta": etiqueta_mas_comun,
                "error_promedio": error_promedio,
            })

            ms_actual = ms_limite

            if etiqueta_mas_comun in etiquetas_interes:
                print(f"🟢 {etiqueta_mas_comun.upper():<10} | {int(ms_inicio)} s → {int(ms_final)} s")
                resultados_ventana_posicion[articulacion].append({
                    "label": etiqueta_mas_comun.upper(),
                    "s_inicio": int(ms_inicio),
                    "s_final": int(ms_final),
                })

    return resultados_ventana_posicion


def quaternion_to_euler_difs(q):
    if q is None or len(q) != 4 or np.any(np.isnan(q)) or np.linalg.norm(q) < 1e-6:
        q = np.array([0, 0, 0, 1])
    try:
        r = Rotation.from_quat(q)
        euler = r.as_euler('xyz', degrees=False)  # radianes
        return euler
    except Exception:
        return np.array([0.0, 0.0, 0.0])


def generar_diccionario_diferencias_angulares(dic_novato, dic_experto):
    dic_diferencias_angulares = {}

    # Inicializar el scaler para las diferencias angulares y error
    scaler = MinMaxScaler(feature_range=(0, 1))

    for articulacion in dic_novato:

        if articulacion not in dic_experto["data"]:
            continue
        if "WROT" not in dic_novato[articulacion] or "wrot" not in dic_experto["data"][articulacion]:
            continue  # Articulación sin cuaterniones

        try:
            # Cuaterniones del novato
            x_n = dic_novato[articulacion]["XROT"]
            y_n = dic_novato[articulacion]["YROT"]
            z_n = dic_novato[articulacion]["ZROT"]
            w_n = dic_novato[articulacion]["WROT"]
            sag_n = dic_novato[articulacion]["sagital"]
            fro_n = dic_novato[articulacion]["frontal"]
            tra_n = dic_novato[articulacion]["transversal"]
            # Cuaterniones del experto
            x_e = dic_experto["data"][articulacion]["xrot"]
            y_e = dic_experto["data"][articulacion]["yrot"]
            z_e = dic_experto["data"][articulacion]["zrot"]
            w_e = dic_experto["data"][articulacion]["wrot"]

            dx_list, dy_list, dz_list, error_list, tiempo_ms, frames = [], [], [], [], [], []

            for i in range(len(dic_novato[articulacion]["tiempo_ms"])):
                q_n = [x_n[i], y_n[i], z_n[i], w_n[i]]
                q_e = [x_e[i], y_e[i], z_e[i], w_e[i]]
                
                euler_n = quaternion_to_euler_difs(q_n)  # [x, y, z]
                euler_e = quaternion_to_euler_difs(q_e)

                dx = euler_e[0] - euler_n[0]
                dy = euler_e[1] - euler_n[1]
                dz = euler_e[2] - euler_n[2]
                error = np.sqrt(dx**2 + dy**2 + dz**2)

                tiempo_ms.append(dic_novato[articulacion]["tiempo_ms"][i])
                frames.append(dic_novato[articulacion]["frames"][i])

                dx_list.append(dx)
                dy_list.append(dy)
                dz_list.append(dz)
                error_list.append(error)

            # Crear el DataFrame con las diferencias angulares
            df_diff = pd.DataFrame({
                "tiempo_ms": tiempo_ms,
                "frames": frames,
                "dxrot": dx_list,
                "dyrot": dy_list,
                "dzrot": dz_list,
                "sagital": sag_n,
                "frontal": fro_n,
                "transversal": tra_n,
                "error": error_list
            })

            # Normalizar las diferencias angulares y el error usando MinMaxScaler
            # Seleccionar las columnas de diferencias angulares y error
            cols_to_scale = ["dxrot", "dyrot", "dzrot", "error"]
            df_diff[cols_to_scale] = scaler.fit_transform(df_diff[cols_to_scale])

            dic_diferencias_angulares[articulacion] = df_diff

        except Exception as e:
            print(f"[ERROR] en {articulacion}: {e}")

    return dic_diferencias_angulares


def etiquetar_por_ventanas_fijas(dic_clusterizado, ventana_ms=1):
    resultados_ventana_angular = {}
    etiquetas_interes = ['muy malo', 'malo', 'regular', 'bueno', 'muy bueno']
    print("Resultado angular por ventana:")

    for articulacion, df in dic_clusterizado.items():
        resultados_ventana_angular[articulacion] = []
        print(f"\n📌 Articulación: {articulacion}")
        if df.empty:
            print("  ⚠️  DataFrame vacío.")
            continue

        df = df.sort_values("tiempo_ms").reset_index(drop=True)
        tiempo_inicio_total = df["tiempo_ms"].min()
        tiempo_fin_total = df["tiempo_ms"].max()

        # Crear ventanas de tamaño fijo
        t_actual = tiempo_inicio_total
        while t_actual < tiempo_fin_total:
            t_final = t_actual + ventana_ms
            ventana_df = df[(df["tiempo_ms"] >= t_actual) & (df["tiempo_ms"] < t_final)]

            if ventana_df.empty:
                t_actual = t_final
                continue
            
            # Contar etiquetas y elegir la más frecuente
            etiquetas = ventana_df["etiqueta"].dropna().tolist()
            if etiquetas:
                etiqueta_dominante = Counter(etiquetas).most_common(1)[0][0]
                if etiqueta_dominante in etiquetas_interes:
                    print(f"🟢 {etiqueta_dominante.upper():<10} | {int(t_actual)} s → {int(t_final)} s")
                    resultados_ventana_angular[articulacion].append({
                    "label": etiqueta_dominante.upper(),
                    "s_inicio": int(t_actual),
                    "s_final": int(t_final),
                    })

            t_actual = t_final
    return resultados_ventana_angular

# Estimación del número de clusters basado en la desviación estándar
def estimar_k_angular_por_std(X, max_clusters=5, threshold_std=0.25):
    std_error = np.std(X)
    #print(f"Desviación estándar del error: {std_error:.5f}")
    
    if std_error < threshold_std:
        #print(f"Se asigna 1 cluster debido a una desviación estándar baja.")
        return 1
    else:
        n_clusters = min(max_clusters, max(2, int(std_error * 50)))
        #print(f"Se estiman {n_clusters} clusters basado en la desviación estándar.")
        return n_clusters


# Etiquetar los clusters según el error angular con umbral global
def etiquetar_clusters_angular_v4(df, col_error="error", col_cluster="cluster", umbral_error=0.05, umbral_error_global=0.15):
    etiquetas = {}
    promedios = df.groupby(col_cluster)[col_error].mean().sort_values()
    error_medio_global = df[col_error].mean()
    
    #print(f"Error medio global: {error_medio_global:.5f}")

    # Definir etiquetas según el error medio global
    if error_medio_global > umbral_error_global:
        etiquetas_posibles = ['muy malo', 'malo', 'regular', 'bueno']
    elif error_medio_global < umbral_error:
        etiquetas_posibles = ['muy bueno', 'bueno', 'regular']
    else:
        etiquetas_posibles = ['muy bueno', 'bueno', 'regular', 'malo']

    # Asegurarse de no repetir etiquetas
    etiquetas_asignadas = set()
    for i, (cluster, error_prom) in enumerate(promedios.items()):
        if i < len(etiquetas_posibles):
            etiqueta_asignada = etiquetas_posibles[i]
            # Si la etiqueta ya está asignada, buscaremos una nueva sin repetir
            while etiqueta_asignada in etiquetas_asignadas:
                etiqueta_asignada = etiquetas_posibles[(i + 1) % len(etiquetas_posibles)]
            etiquetas[cluster] = etiqueta_asignada
            etiquetas_asignadas.add(etiqueta_asignada)
        else:
            # Etiqueta por defecto si hay más clusters que etiquetas
            etiquetas[cluster] = 'bueno'

    return etiquetas


# Clustering y etiquetado por articulación
def agregar_clusters_por_error_angular_v2(dic_diferencias_angulares, max_clusters=5, std_threshold=0.02, random_state=42):
    dic_clusterizado = {}

    for articulacion, df in dic_diferencias_angulares.items():
        #print(f"Procesando articulación: {articulacion}")

        df_clusterizado = df.copy()

        if "error" not in df_clusterizado.columns:
            df_clusterizado["error"] = np.sqrt(df_clusterizado["dxrot"]**2 +
                                               df_clusterizado["dyrot"]**2 +
                                               df_clusterizado["dzrot"]**2)

        n_nan = df_clusterizado["error"].isna().sum()
        if n_nan > 0:
            print(f"[AVISO] {articulacion}: {n_nan} NaNs encontrados. Se eliminarán filas.")
            df_clusterizado = df_clusterizado.dropna(subset=["error"])

        X = df_clusterizado[["error"]].values

        if len(X) < 3:
            print(f"[ADVERTENCIA] {articulacion}: muy pocos datos para clusterizar. Se omitirá.")
            continue

        n_clusters = estimar_k_angular_por_std(X, max_clusters=max_clusters, threshold_std=std_threshold)

        #print(f"Realizando clustering con {n_clusters} clusters.")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)

        df_clusterizado["cluster"] = labels

        cluster_means = df_clusterizado.groupby("cluster")["error"].mean().sort_values()
        label_map = {old: new for new, old in enumerate(cluster_means.index)}
        df_clusterizado["cluster"] = df_clusterizado["cluster"].map(label_map)

        # Etiquetar los clusters usando la nueva función
        etiquetas = etiquetar_clusters_angular_v4(
            df_clusterizado,
            umbral_error=0.05,
            umbral_error_global=0.3
        )

        df_clusterizado["etiqueta"] = df_clusterizado["cluster"].map(etiquetas)

        #print(f"Etiquetas asignadas a los clusters: {etiquetas}")

        dic_clusterizado[articulacion] = df_clusterizado

    return dic_clusterizado