import numpy as np

def seleccionar_articulacion_dict(user_dict, opcion: int):
    joint_names = list(user_dict["data"].keys())

    while True:
        try: 
            if 0 <= opcion < len(joint_names):
                return joint_names[opcion]
            else:
                print("Número fuera de rango. Intenta de nuevo.")
        except ValueError:
            print("Entrada inválida. Ingresa un número válido.")

def calcular_umbral_movimiento_dict(datos, method="posicion", factor=3):
    posiciones = np.column_stack([datos["x"], datos["y"], datos["z"]])

    if method == "velocidad":
        posiciones = np.diff(posiciones, axis=0)

    std_ruido = np.std(posiciones[:100], axis=0)
    umbral = factor * np.mean(std_ruido)

    return umbral

def detectar_inicio_movimiento_dict(datos, umbral):
    posiciones = np.column_stack([datos["x"], datos["y"], datos["z"]])
    diferencias = np.linalg.norm(np.diff(posiciones, axis=0), axis=1)
    frames_supera_umbral = np.where(diferencias > umbral)[0]

    return frames_supera_umbral[0] if len(frames_supera_umbral) > 0 else None

def recortar_dict(user_dict, frame_inicio):
    recortado = {"data": {}, "info": {}}
    for key, array in user_dict["info"].items():
        recortado["info"][key] = array[frame_inicio:]
    for joint, datos in user_dict["data"].items():
        recortado["data"][joint] = {k: v[frame_inicio:] if v is not None else None for k, v in datos.items()}
    return recortado

def igualar_longitud_dicts(user1_dict, user2_dict):
    n_frames1 = len(user1_dict["info"]["frames"])
    n_frames2 = len(user2_dict["info"]["frames"])
    min_frames = min(n_frames1, n_frames2)

    def cortar(dic):
        return {
            "info": {k: v[:min_frames] for k, v in dic["info"].items()},
            "data": {
                joint: {k: v[:min_frames] if v is not None else None for k, v in datos.items()}
                for joint, datos in dic["data"].items()
            }
        }

    return cortar(user1_dict), cortar(user2_dict)

def sync_two_users_dict(user_dict1, user_dict2, joint_id):

    articulacion = seleccionar_articulacion_dict(user_dict1, joint_id)
    print(f"Has seleccionado: {articulacion}")

    if articulacion in user_dict1["data"] and articulacion in user_dict2["data"]:
        umbral1 = calcular_umbral_movimiento_dict(user_dict1["data"][articulacion])
        umbral2 = calcular_umbral_movimiento_dict(user_dict2["data"][articulacion])

        inicio1 = detectar_inicio_movimiento_dict(user_dict1["data"][articulacion], umbral1)
        inicio2 = detectar_inicio_movimiento_dict(user_dict2["data"][articulacion], umbral2)

        if inicio1 is not None and inicio2 is not None:
            dict1_recortado = recortar_dict(user_dict1, inicio1)
            dict2_recortado = recortar_dict(user_dict2, inicio2)
            dict1_final, dict2_final = igualar_longitud_dicts(dict1_recortado, dict2_recortado)

            print(f"Frames finales (user_one): {len(dict1_final['info']['frames'])}")
            print(f"Inicio (ms): {dict1_final['info']['ms'][0]}, Fin (ms): {dict1_final['info']['ms'][-1]}")

            print(f"Frames finales (user_two): {len(dict2_final['info']['frames'])}")
            print(f"Inicio (ms): {dict2_final['info']['ms'][0]}, Fin (ms): {dict2_final['info']['ms'][-1]}")

            return dict1_final, dict2_final
        else:
            print("No se detectó un inicio claro del movimiento.")
            return None, None
    else:
        print(f"La articulación '{articulacion}' no está en ambos diccionarios.")
        return None, None

def alinear_usuarios_procrustes(user1_dict, user2_dict):
    """
    Alinea el segundo usuario al primero usando alineación rígida 3D (sin escalado) en el frame 0.
    """
    # 1. Extraer posiciones de articulaciones comunes
    articulaciones_comunes = list(set(user1_dict["data"].keys()) & set(user2_dict["data"].keys()))
    pos1 = []
    pos2 = []

    for joint in articulaciones_comunes:
        d1 = user1_dict["data"][joint]
        d2 = user2_dict["data"][joint]
        if all(k in d1 for k in ["x", "y", "z"]) and all(k in d2 for k in ["x", "y", "z"]):
            p1 = np.array([d1["x"][0], d1["y"][0], d1["z"][0]])
            p2 = np.array([d2["x"][0], d2["y"][0], d2["z"][0]])
            if not (np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or np.any(np.isinf(p1)) or np.any(np.isinf(p2))):
                pos1.append(p1)
                pos2.append(p2)


    X = np.stack(pos2)  # usuario 2 (el que vamos a mover)
    Y = np.stack(pos1)  # usuario 1 (referencia)

    # 2. Centrar en el centroide
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # 3. Resolver mejor rotación
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    R_opt = Vt.T @ U.T
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T

    # 4. Traslación final
    t = Y_mean - R_opt @ X_mean

    # 5. Aplicar transformación a TODOS los frames del usuario 2
    user2_alineado = {"info": user2_dict["info"].copy(), "data": {}}
    for joint, datos in user2_dict["data"].items():
        if all(k in datos for k in ["x", "y", "z"]):
            pos = np.stack([datos["x"], datos["y"], datos["z"]], axis=1)  # (n_frames, 3)
            pos_rotada = (R_opt @ (pos.T)).T + t  # aplicar rotación y traslación
            user2_alineado["data"][joint] = {
                "x": pos_rotada[:, 0],
                "y": pos_rotada[:, 1],
                "z": pos_rotada[:, 2],
                "xrot": datos.get("xrot"),
                "yrot": datos.get("yrot"),
                "zrot": datos.get("zrot"),
                "wrot": datos.get("wrot")
            }

    return user2_alineado
