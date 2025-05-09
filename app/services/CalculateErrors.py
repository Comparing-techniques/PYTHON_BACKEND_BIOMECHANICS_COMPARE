import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.transform import Rotation as R


def generar_diccionario_diferencias(dic_novato, dic_experto):
    dic_diferencias = {}

    for articulacion in dic_novato["data"]:
        if articulacion not in dic_experto["data"]:
            continue

        x_n, y_n, z_n = dic_novato["data"][articulacion]["x"], dic_novato["data"][articulacion]["y"], dic_novato["data"][articulacion]["z"]
        x_e, y_e, z_e = dic_experto["data"][articulacion]["x"], dic_experto["data"][articulacion]["y"], dic_experto["data"][articulacion]["z"]

        # Creamos diferencias con np.where, asignando NaN donde hay datos faltantes
        dx = np.where(~(np.isnan(x_n) | np.isnan(x_e)), x_e - x_n, np.nan)
        dy = np.where(~(np.isnan(y_n) | np.isnan(y_e)), y_e - y_n, np.nan)
        dz = np.where(~(np.isnan(z_n) | np.isnan(z_e)), z_e - z_n, np.nan)

        error = np.sqrt(dx**2 + dy**2 + dz**2)
        # Normalizar el error usando MinMaxScaler
        scaler = MinMaxScaler()
        error_normalizado = scaler.fit_transform(error.reshape(-1, 1))  # Escalamos la columna error

        ms = dic_novato["info"]["ms"]
        frame = dic_novato["info"]["frames"]

        df_diff = pd.DataFrame({
            "ms": ms,
            "frames": frame,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "error": error_normalizado.flatten()   
        })

        nans = df_diff.isna().sum().sum()
        if nans > 0:
            print(f"[AVISO] {articulacion} tiene {nans} NaNs en diferencias, se mantendrán para conservar sincronía.")

        dic_diferencias[articulacion] = df_diff

    return dic_diferencias


def quaternion_to_euler_difs(q):
    if q is None or len(q) != 4 or np.any(np.isnan(q)) or np.linalg.norm(q) < 1e-6:
        q = np.array([0, 0, 0, 1])
    try:
        r = R.from_quat(q)
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

def quaternion_to_euler_movement(w, x, y, z, degrees=True, order='xyz', debug=False):
    """
    Convierte listas de cuaterniones (w, x, y, z) a ángulos de Euler (en grados por default).
    Devuelve una tupla con tres arrays: (eje_1, eje_2, eje_3) según el orden dado.
    """
    try:
        quats = np.column_stack([x, y, z, w])  # Formato [x, y, z, w]
        norms = np.linalg.norm(quats, axis=1)
        quats[norms < 1e-2] = [0, 0, 0, 1]  # Evita divisiones por 0
        r = R.from_quat(quats)
        euler = r.as_euler(order, degrees=degrees)
        
        if debug:
            print("\n▶ [DEBUG quaternion_to_euler_movement]")
            print("Primeros 5 cuaterniones corregidos:\n", quats[:5])
            print("Primeros 5 ángulos Euler:\n", euler[:5])

        return euler[:, 0], euler[:, 1], euler[:, 2]  # Una tupla con 3 arrays
    except Exception as e:
        print(f"\n⚠️ Error en quaternion_to_euler_movement: {e}")
        n = len(w) if hasattr(w, '__len__') else 1
        return (np.zeros(n), np.zeros(n), np.zeros(n))

def quaternion_to_euler(q):
    """
    Convierte un solo cuaternión (w, x, y, z) a ángulos de Euler en radianes.
    """
    if q is None or len(q) != 4 or np.any(np.isnan(q)) or np.linalg.norm(q) < 1e-6:
        q = np.array([0, 0, 0, 1])
    try:
        r = R.from_quat(q)
        euler = r.as_euler('xyz', degrees=False)  # en radianes
        return {"sagital": euler[0], "frontal": euler[1], "transversal": euler[2]}
    except Exception:
        return {"sagital": 0.0, "frontal": 0.0, "transversal": 0.0}

def crear_dataframes_movimientos(user_dict, rangos_dict, debug=False):
    """
    Crea un diccionario de DataFrames, uno por articulación, que contiene las etiquetas de movimiento por frame.
    """
    dataframes_por_articulacion = {}

    for articulacion, datos in user_dict["data"].items():
        num_frames = len(user_dict["info"]["frames"])

        if "wrot" not in datos or datos["wrot"] is None or len(datos["wrot"]) == 0:
            # Crear DataFrame vacío si no hay rotaciones
            df = pd.DataFrame({
                "frames": user_dict["info"]["frames"],
                "tiempo_ms": user_dict["info"]["ms"],
                "sagital": [None] * num_frames,
                "frontal": [None] * num_frames,
                "transversal": [None] * num_frames
            })
            dataframes_por_articulacion[articulacion] = df
            continue

        # Convertir cuaterniones a ángulos de Euler
        x_angle, y_angle, z_angle = quaternion_to_euler_movement(
            datos["wrot"], datos["xrot"], datos["yrot"], datos["zrot"], degrees=True, order='xyz', debug=debug
        )

        if debug:
            print(f"\n▶ [DEBUG crear_dataframes_movimientos] Articulación: {articulacion}")
            print("Ángulo sagital (X) primeros 5:", x_angle[:5])
            print("Ángulo frontal (Y) primeros 5:", y_angle[:5])
            print("Ángulo transversal (Z) primeros 5:", z_angle[:5])

        etiquetas_sagital = []
        etiquetas_frontal = []
        etiquetas_transversal = []

        rangos = rangos_dict.get(articulacion, {})

        for i in range(num_frames):
            etiqueta_sagital = etiqueta_frontal = etiqueta_transversal = None

            for plano, angulo, etiquetas in zip(
                ["sagital", "frontal", "transversal"],
                [x_angle[i], y_angle[i], z_angle[i]],
                [etiquetas_sagital, etiquetas_frontal, etiquetas_transversal]
            ):
                # print("Entro en frame: ", i, "plano:", plano, "angulo:", angulo)
                movimientos = rangos.get(plano, {})
                for movimiento, (min_val, max_val) in movimientos.items():
                    if min_val <= angulo <= max_val:
                        if plano == "sagital":
                            etiqueta_sagital = movimiento
                        elif plano == "frontal":
                            etiqueta_frontal = movimiento
                        elif plano == "transversal":
                            etiqueta_transversal = movimiento
                        break

            etiquetas_sagital.append(etiqueta_sagital)
            etiquetas_frontal.append(etiqueta_frontal)
            etiquetas_transversal.append(etiqueta_transversal)

        df = pd.DataFrame({
            "frames": user_dict["info"]["frames"],
            "tiempo_ms": user_dict["info"]["ms"],
            "XROT": datos["xrot"],
            "YROT": datos["yrot"],
            "ZROT": datos["zrot"],
            "WROT": datos["wrot"],
            "sagital": etiquetas_sagital,
            "frontal": etiquetas_frontal,
            "transversal": etiquetas_transversal
        })

        dataframes_por_articulacion[articulacion] = df

    return dataframes_por_articulacion

# Diccionario de rangos de movimiento a partir de cuaterniones convertidos a ángulos (en grados)
# Se utiliza la siguiente convención de signos:
#   - En el plano sagital:
#       * Flexión: de 0° a un valor positivo (ej. 0° a +80°).
#       * Extensión: de un valor negativo a 0° (ej. -70° a 0°).
#   - En el plano frontal:
#       * La dirección positiva indica, por ejemplo, desviación radial o abducción,
#         mientras que la dirección negativa indica desviación cubital o aducción.
#   - En el plano transversal:
#       * Rotación externa se define con valores positivos y rotación interna con negativos.
#   - En movimientos multiplanares, se permite un rango simétrico (por ejemplo, circunducción).

rangos_movimientos_con_signos = {
    "Articulación Atlanto-Occipital y Atlanto-Axial (AO-AA)": {
        "sagital": {
            "Flexión": [0, 25],          # 0° a +25° indica flexión
            "Extensión": [-15, 0]        # -15° a 0° indica extensión
        },
        "frontal": {
            "Inclinación Lateral izquierda": [-20, 0],  # -20° (inclinación hacia un lado) a +20° (hacia el otro)
            "Inclinación Lateral derecha": [0, 20]
        },
        "transversal": {
            "Rotación interna": [-80, 0],
            "Rotacion externa": [0, 80]       # -80° para rotación interna y +80° para rotación externa
        }
    },
    "Columna Cervical": {
        "sagital": {
            "Flexión": [0, 50],
            "Extensión": [-60, 0]
        },
        "frontal": {
            "Inclinación Lateral izquierda": [-45, 0],
            "Inclinación Lateral derecha": [0, 45]
        },
        "transversal": {
            "Rotación interna": [-80, 0],
            "Rotacion externa": [0, 80]
        }
    },
    "Columna Torácica y Lumbar": {
        "sagital": {
            "Flexión": [0, 30],
            "Extensión": [-30, 0]
        },
        "frontal": {
            "Inclinación Lateral izquierda": [-20, 0],
            "Inclinación Lateral derecha": [0, 20]
        },
        "transversal": {
            "Rotación interna": [-25, 0],
            "Rotacion externa": [0, 25]
        }
    },
    "Articulación Sacroilíaca": {
        "sagital": {
            "Nutación": [0, 10],        # Nutación: 0° a +10°
            "Contranutación": [-10, 0]   # Contranutación: -10° a 0°
        },
        "transversal": {
            "Ligera Rotación izquierda": [-5, 0],
            "Ligera Rotación derecha": [0, 5]
        }
    },
    "Cadera Izquierda (Coxofemoral)": {
        "sagital": {
            "Flexión": [0, 120],
            "Extensión": [-30, 0]
        },
        "frontal": {
            "Abducción": [0, 45],       # Abducción: 0° a +45°
            "Aducción": [-30, 0]        # Aducción: -30° a 0°
        },
        "transversal": {
            "Rotación Interna": [-40, 0],   # Rotación interna: -40° a 0°
            "Rotación Externa": [0, 50]      # Rotación externa: 0° a +50°
        }
    },
    "Cadera Derecha (Coxofemoral)": {
        "sagital": {
            "Flexión": [0, 120],
            "Extensión": [-30, 0]
        },
        "frontal": {
            "Abducción": [0, 45],
            "Aducción": [-30, 0]
        },
        "transversal": {
            "Rotación Interna": [-40, 0],
            "Rotación Externa": [0, 50]
        }
    },
    "Rodilla Izquierda": {
        "sagital": {
            "Flexión": [0, 130],
            "Extensión": [-5, 0]
        },
        "transversal": {
            "Rotación Interna (leve)": [-10, 0],
            "Rotación Externa (leve)": [0, 10]
        }
    },
    "Rodilla Derecha": {
        "sagital": {
            "Flexión": [0, 130],
            "Extensión": [-5, 0]
        },
        "transversal": {
            "Rotación Interna (leve)": [-10, 0],
            "Rotación Externa (leve)": [0, 10]
        }
    },
    "Tobillo Izquierdo (Art. Talocrural)": {
        "sagital": {
            "Flexión Dorsal": [0, 20],    # Dorsiflexión: 0° a +20°
            "Flexión Plantar": [-50, 0]   # Plantarflexión: -50° a 0°
        },
        "frontal": {
            "Inversión": [0, 35],         # Inversión: 0° a +35°
            "Eversión": [-25, 0]          # Eversión: -25° a 0°
        }
    },
    "Tobillo Derecho (Art. Talocrural)": {
        "sagital": {
            "Flexión Dorsal": [0, 20],
            "Flexión Plantar": [-50, 0]
        },
        "frontal": {
            "Inversión": [0, 35],
            "Eversión": [-25, 0]
        }
    },
    "Articulaciones del Pie Izquierdo (Subtalar y MTF)": {
        "sagital": {
            "Flexión": [0, 15],
            "Extensión": [-15, 0]
        },
        "frontal": {
            "Inversión": [0, 30],
            "Eversión": [-20, 0]
        }
    },
    "Articulaciones del Pie Derecho (Subtalar y MTF)": {
        "sagital": {
            "Flexión": [0, 15],
            "Extensión": [-15, 0]
        },
        "frontal": {
            "Inversión": [0, 30],
            "Eversión": [-20, 0]
        }
    },
    "Articulación Esternoclavicular": {
        "sagital": {
            "Elevación": [0, 30],
            "Depresión": [-10, 0]
        },
        "frontal": {
            "Protracción": [0, 20],
            "Retracción": [-20, 0]
        },
        "transversal": {
            "Rotación": [-15, 15]  # Rotación con valores negativos y positivos
        }
    },
    "Articulación Acromioclavicular": {
        "sagital": {
            "Elevación": [0, 25],
            "Depresión": [-10, 0]
        },
        "transversal": {
            "Rotación Interna": [-15, 0],
            "Rotación Externa": [0, 15]
        }
    },
    "Articulación Escapulotorácica": {
        "sagital": {
            "Deslizamiento Superior": [0, 20],
            "Deslizamiento Inferior": [-20, 0]
        },
        "transversal": {
            "Rotación Superior": [0, 30],
            "Rotación Inferior": [-30, 0]
        }
    },
    "Hombro Izquierdo (Glenohumeral)": {
        "sagital": {
            "Flexión": [0, 180],
            "Extensión": [-60, 0]
        },
        "frontal": {
            "Abducción": [0, 180],
            "Aducción": [-50, 0]
        },
        "transversal": {
            "Rotación Interna": [-70, 0],
            "Rotación Externa": [0, 90]
        },
        "multiplanar": {
            "Circunducción": [-180, 180]  # Rango completo para movimientos circulares
        }
    },
    "Hombro Derecho (Glenohumeral)": {
        "sagital": {
            "Flexión": [0, 180],
            "Extensión": [-60, 0]
        },
        "frontal": {
            "Abducción": [0, 180],
            "Aducción": [-50, 0]
        },
        "transversal": {
            "Rotación Interna": [-70, 0],
            "Rotación Externa": [0, 90]
        },
        "multiplanar": {
            "Circunducción": [-180, 180]
        }
    },
    "Codo Izquierdo": {
        "sagital": {
            "Flexión": [0, 150],
            "Extensión": [-10, 0]
        },
        "transversal": {
            "Pronación": [0, 90],   # Pronación: 0° a +90°
            "Supinación": [-90, 0]   # Supinación: -90° a 0°
        }
    },
    "Codo Derecho": {
        "sagital": {
            "Flexión": [0, 150],
            "Extensión": [-10, 0]
        },
        "transversal": {
            "Pronación": [0, 90],
            "Supinación": [-90, 0]
        }
    },
    "Muñeca Izquierda": {
        "sagital": {
            "Flexión": [0, 80],
            "Extensión": [-70, 0]
        },
        "frontal": {
            "Desviación Radial": [0, 20],   # Desviación radial: 0° a +20°
            "Desviación Cubital": [-30, 0]    # Desviación cubital: -30° a 0°
        }
    },
    "Muñeca Derecha": {
        "sagital": {
            "Flexión": [0, 80],
            "Extensión": [-70, 0]
        },
        "frontal": {
            "Desviación Radial": [0, 20],
            "Desviación Cubital": [-30, 0]
        }
    }
}