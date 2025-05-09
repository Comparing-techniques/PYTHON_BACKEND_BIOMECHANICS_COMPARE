import numpy as np
from scipy.spatial.transform import Rotation as R

def promedio_posicion(marker1, marker2, data_dict, step=1):
    """
    Calcula el promedio de posición (x, y, z) entre dos marcadores usando solo NumPy arrays.

    Args:
        marker1 (str): Nombre del primer marcador.
        marker2 (str): Nombre del segundo marcador.
        data_dict (dict): Diccionario con estructura {"data": ..., "info": ...}.
        step (int): Intervalo de submuestreo.

    Retorna:
        dict: Diccionario con claves 'x', 'y', 'z', 'frame', 'ms' conteniendo arrays promedio.
    """
    d1 = data_dict["data"][marker1]
    d2 = data_dict["data"][marker2]

    x_avg = (np.array(d1["x"]) + np.array(d2["x"])) / 2
    y_avg = (np.array(d1["y"]) + np.array(d2["y"])) / 2
    z_avg = (np.array(d1["z"]) + np.array(d2["z"])) / 2

    # Submuestreo si se pide
    x_avg = x_avg[::step]
    y_avg = y_avg[::step]
    z_avg = z_avg[::step]

    return{
        "x": np.array(x_avg),
        "y": np.array(y_avg),
        "z": np.array(z_avg),
    }

def replace_zero_norm_quat(q, tol=1e-10):
    """
    Reemplaza cuaterniones con norma cercana a cero o inválidos (NaN, inf) por el cuaternión identidad (0, 0, 0, 1).
    """
    if q is None:
        raise ValueError("Cuaterniones no pueden ser None")

    q = np.asarray(q).copy()

    # Detectamos cuaterniones con valores inválidos
    invalid_mask = np.isnan(q).any(axis=1) | np.isinf(q).any(axis=1)

    # Calculamos norma y detectamos cuaterniones con norma baja
    norms = np.linalg.norm(q, axis=1)
    zero_norms = norms < tol

    # Combinamos ambas condiciones
    to_replace = zero_norms | invalid_mask

    q[to_replace] = np.array([0, 0, 0, 1])
    return q


def calcular_orientacion_segmento(markers, data_dict, step=1):
    """
    Calcula la orientación de un segmento usando uno o dos marcadores.
    Si hay un marcador y tiene cuaternión, lo devuelve directamente.
    Si hay dos, calcula la orientación relativa o estimada con vectores.

    Args:
        markers (list): Lista con 1 o 2 nombres de marcadores.
        data_dict (dict): Diccionario con la estructura explicada (data/info).
        step (int): Submuestreo para reducir cantidad de frames.

    Retorna:
        np.ndarray: Arreglo (N, 4) de cuaterniones [xrot, yrot, zrot, wrot].
    """

    def get_quat(marker):
        """Extrae y normaliza los cuaterniones de un marcador."""
        d = data_dict["data"].get(marker)
        if d is None:
            raise ValueError(f"Marcador '{marker}' no encontrado en los datos")

        has_quat = d.get("isQuat", False) and all(k in d for k in ['xrot', 'yrot', 'zrot', 'wrot'])
        if has_quat:
            q = np.stack([d["xrot"], d["yrot"], d["zrot"], d["wrot"]], axis=1)
            q = replace_zero_norm_quat(q)
            return q[::step]
        else:
            # Si no hay rotación, se asume identidad para cada frame
            n = len(d["x"])
            return np.tile(np.array([[0, 0, 0, 1]]), (n//step, 1))

    def get_position(marker):
        """Extrae posiciones (x, y, z) de un marcador."""
        d = data_dict["data"].get(marker)
        if d is None:
            raise ValueError(f"Marcador '{marker}' no encontrado en los datos")
        return np.stack([d["x"], d["y"], d["z"]], axis=1)[::step]

    if len(markers) == 1:
        return get_quat(markers[0])

    elif len(markers) == 2:
        m1, m2 = markers
        d1 = data_dict["data"].get(m1)
        d2 = data_dict["data"].get(m2)
        if d1 is None or d2 is None:
            raise ValueError(f"Marcadores '{m1}' o '{m2}' no encontrados en los datos")

        # Si ambos marcadores tienen cuaternión, usamos su diferencia relativa
        if d1.get("isQuat", False) and d2.get("isQuat", False):
            q1 = get_quat(m1)
            q2 = get_quat(m2)
            R1 = R.from_quat(q1)
            R2 = R.from_quat(q2)
            Rseg = R2 * R1.inv()
            return Rseg.as_quat()
        else:
            # Si no hay cuaterniones, estimamos la orientación usando vectores
            pos1 = get_position(m1)
            pos2 = get_position(m2)
            vec = pos2 - pos1

            norms = np.linalg.norm(vec, axis=1)
            norms[norms < 1e-8] = 1
            vec_norm = vec / norms[:, np.newaxis]

            ref = np.array([1, 0, 0])  # Vector de referencia
            dot = np.einsum('ij,j->i', vec_norm, ref)
            cross = np.cross(ref, vec_norm)

            angles = np.arccos(np.clip(dot, -1, 1))
            axes_norms = np.linalg.norm(cross, axis=1)
            axes = np.divide(cross, axes_norms[:, np.newaxis], where=axes_norms[:, np.newaxis] != 0)
            axes[axes_norms == 0] = [0, 0, 1]

            rotations = R.from_rotvec(axes * angles[:, np.newaxis])
            return rotations.as_quat()

    else:
        raise ValueError("La lista de marcadores debe tener 1 o 2 elementos.")

def calcular_cinematic_rel(seg_sup_dict, seg_inf_dict=None, user_dict=None, step=1, order="inferior_to_superior"):
    """
    Calcula la cinemática relativa entre dos segmentos (orientación relativa).
    Si solo se da uno, devuelve la orientación de ese segmento.

    Args:
        seg_sup_dict (dict): Segmento superior (tiene 'inicio' y 'fin').
        seg_inf_dict (dict): Segmento inferior (opcional).
        user_dict (dict): Diccionario con estructura 'data' y 'info'.
        step (int): Submuestreo temporal.
        order (str): Dirección de comparación: 'inferior_to_superior' o 'superior_to_inferior'.

    Retorna:
        np.ndarray: Cuaterniones relativos (N, 4).
    """

    if user_dict is None:
        raise ValueError("Se requiere el parámetro 'user_dict'")

    # Si solo hay un segmento, se devuelve su orientación absoluta
    if seg_inf_dict is None:
        markers_sup = [seg_sup_dict["inicio"], seg_sup_dict["fin"]]
        return calcular_orientacion_segmento(markers_sup, user_dict, step=step)

    # Extraer orientaciones de ambos segmentos
    markers_sup = [seg_sup_dict["inicio"], seg_sup_dict["fin"]]
    markers_inf = [seg_inf_dict["inicio"], seg_inf_dict["fin"]]

    orient_sup = calcular_orientacion_segmento(markers_sup, user_dict, step=step)
    orient_inf = calcular_orientacion_segmento(markers_inf, user_dict, step=step)

    # Convertir a objetos de rotación
    R_sup = R.from_quat(replace_zero_norm_quat(orient_sup))
    R_inf = R.from_quat(replace_zero_norm_quat(orient_inf))

    # Calcular la rotación relativa según el orden especificado
    if order == "inferior_to_superior":
        R_rel = R_inf * R_sup.inv()
    elif order == "superior_to_inferior":
        R_rel = R_sup.inv() * R_inf
    else:
        raise ValueError("El parámetro 'order' debe ser 'inferior_to_superior' o 'superior_to_inferior'.")

    return R_rel.as_quat()

def create_dict_with_data(user_dict):
    joints_calculations_mod = {
        "Articulación Atlanto-Occipital y Atlanto-Axial (AO-AA)": {
            "movimientos": ["Flexión", "Extensión", "Inclinación Lateral", "Rotación"],
            "marcadores": ["Neck", "C7", "LFHD", "LBHD", "RBHD", "RFHD"],
            "segmentos": {
                "s_superior": {"inicio": "LFHD", "fin": "C7"},
                "s_inferior": {"inicio": "Neck", "fin": "C7"}
            },
            "centro": promedio_posicion("C7" ,"Neck", user_dict)
        },
        "Columna Cervical": {
            "movimientos": ["Flexión", "Extensión", "Inclinación Lateral", "Rotación"],
            "marcadores": ["Neck", "C7", "T10", "CLAV", "STRN", "RBAK"],
            "segmentos": {
                "s_superior": {"inicio": "Neck", "fin": "C7"},
                "s_inferior": {"inicio": "T10", "fin": "RBAK"}
            },
            "centro": promedio_posicion("C7", "T10", user_dict)
        },
        "Columna Torácica y Lumbar": {
            "movimientos": ["Flexión", "Extensión", "Inclinación Lateral", "Rotación"],
            "marcadores": ["T10", "RBAK", "CLAV", "STRN", "Ab", "Chest"],
            "segmentos": {
                "s_superior": {"inicio": "CLAV", "fin": "RBAK"},
                "s_inferior": {"inicio": "T10", "fin": "Chest"}
            },
            "centro": promedio_posicion("RBAK", "T10", user_dict)
        },
        "Articulación Sacroilíaca": {
            "movimientos": ["Nutación", "Contranutación", "Ligera Rotación"],
            "marcadores": ["LASI", "RASI", "LPSI", "RPSI"],
            "segmentos": {
                "s_superior": {"inicio": "LASI", "fin": "RASI"},
                "s_inferior": {"inicio": "LPSI", "fin": "RPSI"}
            },
            "centro": promedio_posicion("RASI", "LPSI", user_dict)
        },
        "Cadera Izquierda (Coxofemoral)": {
            "movimientos": ["Flexión", "Extensión", "Abducción", "Aducción", "Rotación Interna", "Rotación Externa"],
            "marcadores": ["LASI", "RASI", "LPSI", "RPSI", "LThigh", "LTHI", "LKNE"],
            "segmentos": {
                "s_superior": {"inicio": "LASI", "fin": "LPSI"},
                "s_inferior": {"inicio": "LThigh", "fin": "LKNE"}
            },
            "centro": promedio_posicion("LPSI", "LThigh", user_dict)
        },
        "Cadera Derecha (Coxofemoral)": {
            "movimientos": ["Flexión", "Extensión", "Abducción", "Aducción", "Rotación Interna", "Rotación Externa"],
            "marcadores": ["LASI", "RASI", "LPSI", "RPSI", "RThigh", "RTHI", "RKNE"],
            "segmentos": {
                "s_superior": {"inicio": "RASI", "fin": "RPSI"},
                "s_inferior": {"inicio": "RThigh", "fin": "RKNE"}
            },
            "centro": promedio_posicion("RPSI", "RThigh", user_dict)
        },
        "Rodilla Izquierda": {
            "movimientos": ["Flexión", "Extensión", "Rotación Interna (leve)", "Rotación Externa (leve)"],
            "marcadores": ["LThigh", "LTHI", "LKNE", "LShin", "LTIB"],
            "segmentos": {
                "s_superior": {"inicio": "LThigh", "fin": "LKNE"},
                "s_inferior": {"inicio": "LShin", "fin": "LTIB"}
            },
            "centro": promedio_posicion("LKNE", "LShin", user_dict)
        },
        "Rodilla Derecha": {
            "movimientos": ["Flexión", "Extensión", "Rotación Interna (leve)", "Rotación Externa (leve)"],
            "marcadores": ["RThigh", "RTHI", "RKNE", "RShin", "RTIB"],
            "segmentos": {
                "s_superior": {"inicio": "RThigh", "fin": "RKNE"},
                "s_inferior": {"inicio": "RShin", "fin": "RTIB"}
            },
            "centro": promedio_posicion("RKNE", "RShin", user_dict)
        },
        "Tobillo Izquierdo (Art. Talocrural)": {
            "movimientos": ["Flexión Dorsal", "Flexión Plantar", "Inversión", "Eversión"],
            "marcadores": ["LShin", "LTIB", "LANK", "LFoot", "LToe", "LHEE"],
            "segmentos": {
                "s_superior": {"inicio": "LShin", "fin": "LTIB"},
                "s_inferior": {"inicio": "LANK", "fin": "LHEE"}
            },
            "centro": promedio_posicion("LTIB", "LANK", user_dict)
        },
        "Tobillo Derecho (Art. Talocrural)": {
            "movimientos": ["Flexión Dorsal", "Flexión Plantar", "Inversión", "Eversión"],
            "marcadores": ["RShin", "RTIB", "RANK", "RFoot", "RToe", "RHEE"],
            "segmentos": {
                "s_superior": {"inicio": "RShin", "fin": "RTIB"},
                "s_inferior": {"inicio": "RANK", "fin": "RHEE"}
            },
            "centro": promedio_posicion("RTIB", "RANK", user_dict)
        },
        "Articulaciones del Pie Izquierdo (Subtalar y MTF)": {
            "movimientos": ["Inversión", "Eversión", "Flexión", "Extensión"],
            "marcadores": ["LFoot", "LToe", "LHEE"],
            "segmentos": {
                "s_unico": {"inicio": "LFoot", "fin": "LHEE"}
            },
            "centro": promedio_posicion("LFoot", "LHEE", user_dict)
        },
        "Articulaciones del Pie Derecho (Subtalar y MTF)": {
            "movimientos": ["Inversión", "Eversión", "Flexión", "Extensión"],
            "marcadores": ["RFoot", "RToe", "RHEE"],
            "segmentos": {
                "s_unico": {"inicio": "RFoot", "fin": "RHEE"}
            },
            "centro": promedio_posicion("RFoot", "RHEE", user_dict)
        },
        "Articulación Esternoclavicular": {
            "movimientos": ["Elevación", "Depresión", "Protracción", "Retracción", "Rotación"],
            "marcadores": ["CLAV", "STRN", "RBAK"],
            "segmentos": {
                "s_superior": {"inicio": "CLAV", "fin": "CLAV"},
                "s_inferior": {"inicio": "STRN", "fin": "RBAK"}
            },
            "centro": promedio_posicion("CLAV", "STRN", user_dict)
        },
        "Articulación Acromioclavicular": {
            "movimientos": ["Rotación Interna", "Rotación Externa", "Elevación", "Depresión"],
            "marcadores": ["CLAV", "STRN", "RBAK", "LShoulder", "RShoulder", "LSHO", "RSHO"],
            "segmentos": {
                "s_superior": {"inicio": "LShoulder", "fin": "RSHO"},
                "s_inferior": {"inicio": "CLAV", "fin": "RBAK"}
            },
            "centro": promedio_posicion("RBAK", "LShoulder", user_dict)
        },
        "Articulación Escapulotorácica": {
            "movimientos": ["Deslizamiento Superior", "Deslizamiento Inferior", "Rotación Superior", "Rotación Inferior"],
            "marcadores": ["RBAK", "CLAV", "STRN", "LShoulder", "RShoulder", "LSHO", "RSHO"],
            "segmentos": {
                "s_superior": {"inicio": "LShoulder", "fin": "RSHO"},
                "s_inferior": {"inicio": "RBAK", "fin": "STRN"}
            },
            "centro": promedio_posicion("RSHO", "RBAK", user_dict)
        },
        "Hombro Izquierdo (Glenohumeral)": {
            "movimientos": ["Flexión", "Extensión", "Abducción", "Aducción", "Rotación Interna", "Rotación Externa", "Circunducción"],
            "marcadores": ["LShoulder", "LSHO", "LUArm", "LUPA", "LELB", "CLAV", "STRN"],
            "segmentos": {
                "s_superior": {"inicio": "CLAV", "fin": "LSHO"},
                "s_inferior": {"inicio": "LUArm", "fin": "LELB"}
            },
            "centro": promedio_posicion("LSHO", "LUArm", user_dict)
        },
        "Hombro Derecho (Glenohumeral)": {
            "movimientos": ["Flexión", "Extensión", "Abducción", "Aducción", "Rotación Interna", "Rotación Externa", "Circunducción"],
            "marcadores": ["RShoulder", "RSHO", "RUArm", "RUPA", "RELB", "CLAV", "STRN"],
            "segmentos": {
                "s_superior": {"inicio": "CLAV", "fin": "RSHO"},
                "s_inferior": {"inicio": "RUArm", "fin": "RELB"}
            },
            "centro": promedio_posicion("RSHO", "RUArm", user_dict)
        },
        "Codo Izquierdo": {
            "movimientos": ["Flexión", "Extensión", "Pronación", "Supinación"],
            "marcadores": ["LUArm", "LUPA", "LELB", "LFArm", "LFRM"],
            "segmentos": {
                "s_superior": {"inicio": "LUArm", "fin": "LELB"},
                "s_inferior": {"inicio": "LFArm", "fin": "LFRM"}
            },
            "centro": promedio_posicion("LELB", "LFArm", user_dict)
        },
        "Codo Derecho": {
            "movimientos": ["Flexión", "Extensión", "Pronación", "Supinación"],
            "marcadores": ["RUArm", "RUPA", "RELB", "RFArm", "RFRM"],
            "segmentos": {
                "s_superior": {"inicio": "RUArm", "fin": "RELB"},
                "s_inferior": {"inicio": "RFArm", "fin": "RFRM"}
            },
            "centro": promedio_posicion("RELB", "RFArm", user_dict)
        },
        "Muñeca Izquierda": {
            "movimientos": ["Flexión", "Extensión", "Desviación Radial", "Desviación Cubital"],
            "marcadores": ["LFArm", "LFRM", "LWRB", "LWRA", "LHand"],
            "segmentos": {
                "s_superior": {"inicio": "LFArm", "fin": "LFRM"},
                "s_inferior": {"inicio": "LWRB", "fin": "LHand"}
            },
            "centro": promedio_posicion("LFRM", "LWRB", user_dict)
        },
        "Muñeca Derecha": {
            "movimientos": ["Flexión", "Extensión", "Desviación Radial", "Desviación Cubital"],
            "marcadores": ["RFArm", "RFRM", "RWRB", "RWRA", "RHand"],
            "segmentos": {
                "s_superior": {"inicio": "RFArm", "fin": "RFRM"},
                "s_inferior": {"inicio": "RWRB", "fin": "RHand"}
            },
            "centro": promedio_posicion("RFRM", "RWRB", user_dict)
        }
    }
    return joints_calculations_mod

def convertir_a_dict_articulaciones(user_dict, joints_calculations, step=1):
    articulaciones_data = {}
    tiempos_globales = None

    for articulacion, info in joints_calculations.items():
        centro = info.get("centro", None)
        if centro is None or not isinstance(centro, dict):
            print(f"[DEBUG] Centro no válido para {articulacion}")
            continue

        segmentos = info.get("segmentos", {})
        if "s_superior" in segmentos and "s_inferior" in segmentos:
            sup_seg = segmentos["s_superior"]
            inf_seg = segmentos["s_inferior"]
            resultado_array = calcular_cinematic_rel(sup_seg, inf_seg, user_dict=user_dict, step=step)
        elif "s_unico" in segmentos:
            resultado_array = calcular_cinematic_rel(segmentos["s_unico"], user_dict=user_dict, step=step)
        else:
            print(f"[DEBUG] Segmentos no encontrados para {articulacion}")
            continue

        # Separar los cuaterniones en diccionario
        rot_keys = ['xrot', 'yrot', 'zrot', 'wrot']
        resultado = {}
        if resultado_array.ndim == 2 and resultado_array.shape[1] == 4:
            for i, key in enumerate(rot_keys):
                resultado[key] = resultado_array[:, i]
        else:
            print(f"[DEBUG] Resultado inesperado para {articulacion}: shape={resultado_array.shape}")
            for key in rot_keys:
                resultado[key] = np.full(len(centro['x'][::step]), np.nan)

        # Creamos el subdiccionario para esta articulación
        articulaciones_data[articulacion] = {}

        n = len(centro['x'][::step])  # número de frames después del submuestreo

        # Agregamos posiciones del centro
        for k in ['x', 'y', 'z']:
            articulaciones_data[articulacion][k] = np.array(centro[k])[::step] if k in centro else np.full(n, np.nan)

        # Agregamos rotaciones del resultado
        for k in rot_keys:
            articulaciones_data[articulacion][k] = resultado.get(k, np.full(n, np.nan))

        # Guardamos los tiempos una sola vez
        if tiempos_globales is None:
            tiempos_globales = {
                "frames": user_dict['info']['frames'][::step][:n],
                "ms": user_dict['info']['ms'][::step][:n]
            }
    return {
        "data": articulaciones_data,
        "info": tiempos_globales if tiempos_globales else {"frames": np.array([]), "ms": np.array([])}
    }
