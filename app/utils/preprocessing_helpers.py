import pandas as pd
import numpy as np
from collections import Counter
from scipy.spatial.transform import Rotation as R


# ---------- I/O and parsing ----------
def load_excel_file(file_path: str) -> pd.DataFrame:
    """
    Carga un archivo Excel en un DataFrame.
    """
    try:
        df = pd.read_excel(file_path, header=None)
        return df
    except Exception as e:
        raise RuntimeError(f"Error cargando Excel: {e}")


def extract_numeric_list(df: pd.DataFrame) -> list[float]:
    """
    Extrae lista de valores numéricos bajo la celda 'Time'.
    """
    row_idx = df[df.iloc[:, 1].astype(str).str.contains('Time', na=False)].index
    if row_idx.empty:
        raise ValueError("No se encontró 'Time'")
    start = row_idx[0]
    time_list = []
    for item in df.iloc[start+1:, 1]:
        if pd.isna(item) or item == '':
            break
        s = str(item).replace(',', '.').strip()
        if s.isdigit() and len(s) > 6:
            s = round(int(s) / 1e6, 4)
        try:
            time_list.append(float(s))
        except ValueError:
            continue
    return time_list


def create_tuple_names_list(df: pd.DataFrame) -> list[tuple]:
    """
    Cuenta marcadores declarados bajo 'Name'.
    """
    row_idx = df[df.iloc[:,1] == 'Name'].index
    if row_idx.empty:
        raise ValueError("No se encontró 'Name'")
    row = row_idx[0]
    raw = df.iloc[row,2:].tolist()
    data = [item.split(':',1)[1].strip() if isinstance(item,str) and ':' in item else item
            for item in raw]
    return list(Counter(data).items())


# ---------- Marker dict creation ----------
def creating_marker_dict(df: pd.DataFrame, list_tuples) -> dict:
    time_list = extract_numeric_list(df.copy())
    num_frames = len(time_list)
    frames = list(range(1, num_frames+1))
    out = {'data': {}, 'info': {'ms': time_list, 'frames': frames}}
    for name, count in list_tuples:
        entry = dict(x=None, y=None, z=None,
                     xrot=None, yrot=None, zrot=None,
                     isQuat=(count==7))
        if count==7:
            entry['wrot'] = None
        out['data'][name] = entry
    return out


def add_data_to_dict(df: pd.DataFrame, data_dict: dict, list_tuples: list, start_row: int) -> dict:
    rows = df.iloc[:,2].astype(str)
    idx = df[rows.str.contains('Rotation|Position', na=False)].index
    if idx.empty:
        raise ValueError("No se encontró Rotation/Position")
    base = idx[0]
    name_row = base - 2
    type_row = base + 1
    pos_row = base
    rotation_map = {'X':'xrot','Y':'yrot','Z':'zrot','W':'wrot'}
    position_map = {'X':'x','Y':'y','Z':'z'}
    N = len(data_dict['info']['ms'])
    for col in df.columns[start_row:]:
        name = str(df.iloc[name_row, col]).split(':')[-1].strip()
        coord = str(df.iloc[type_row, col]).strip()
        which = str(df.iloc[pos_row, col]).strip()
        arr = pd.to_numeric(df.iloc[type_row+1:, col], errors='coerce').to_numpy()
        if arr.shape[0] != N:
            continue
        if which=='Rotation' and coord in rotation_map:
            data_dict['data'][name][rotation_map[coord]] = arr
        elif which=='Position' and coord in position_map:
            data_dict['data'][name][position_map[coord]] = arr
    return data_dict


# ---------- Utility computations ----------
def promedio_posicion(marker1: str, marker2: str, data_dict: dict, step: int=1) -> dict:
    d1 = data_dict['data'][marker1]
    d2 = data_dict['data'][marker2]
    x = (np.array(d1['x']) + np.array(d2['x'])) / 2
    y = (np.array(d1['y']) + np.array(d2['y'])) / 2
    z = (np.array(d1['z']) + np.array(d2['z'])) / 2
    return {'x': x[::step], 'y': y[::step], 'z': z[::step]}


def replace_zero_norm_quat(q: np.ndarray, tol: float=1e-10) -> np.ndarray:
    q = np.asarray(q).copy()
    invalid = np.isnan(q).any(axis=1) | np.isinf(q).any(axis=1)
    norm = np.linalg.norm(q, axis=1)
    zero = norm < tol
    mask = invalid | zero
    q[mask] = np.array([0,0,0,1])
    return q


def calcular_orientacion_segmento(markers: list, data_dict: dict, step: int=1) -> np.ndarray:
    def get_quat(m):
        d = data_dict['data'][m]
        if d.get('isQuat') and all(k in d for k in ['xrot','yrot','zrot','wrot']):
            q = np.stack([d['xrot'],d['yrot'],d['zrot'],d['wrot']],axis=1)
            return replace_zero_norm_quat(q)[::step]
        n = len(d['x'])
        return np.tile([0,0,0,1], (n//step,1))
    def get_pos(m):
        d = data_dict['data'][m]
        return np.stack([d['x'],d['y'],d['z']],axis=1)[::step]
    if len(markers)==1:
        return get_quat(markers[0])
    m1,m2 = markers
    d1,d2 = data_dict['data'][m1], data_dict['data'][m2]
    if d1['isQuat'] and d2['isQuat']:
        q1,q2 = get_quat(m1), get_quat(m2)
        R1 = R.from_quat(q1); R2 = R.from_quat(q2)
        return (R2 * R1.inv()).as_quat()
    p1,p2 = get_pos(m1), get_pos(m2)
    vec = p2 - p1
    norms = np.linalg.norm(vec,axis=1)
    norms[norms<1e-8]=1
    vn = vec / norms[:,None]
    ref = np.array([1,0,0])
    dots = np.einsum('ij,j->i', vn, ref)
    crosses = np.cross(ref, vn)
    angles = np.arccos(np.clip(dots,-1,1))
    axnorm = np.linalg.norm(crosses,axis=1)
    axes = np.divide(crosses, axnorm[:,None], where=axnorm[:,None]!=0)
    axes[axnorm==0] = [0,0,1]
    return R.from_rotvec(axes*angles[:,None]).as_quat()


def calcular_cinematic_rel(seg_sup: dict, seg_inf: dict=None, user_dict: dict=None,
                            step: int=1, order: str="inferior_to_superior") -> np.ndarray:
    if user_dict is None:
        raise ValueError("Se requiere user_dict")
    if seg_inf is None:
        return calcular_orientacion_segmento([seg_sup['inicio'], seg_sup['fin']], user_dict, step)
    sup = calcular_orientacion_segmento([seg_sup['inicio'],seg_sup['fin']], user_dict, step)
    inf = calcular_orientacion_segmento([seg_inf['inicio'],seg_inf['fin']], user_dict, step)
    Rsup,Rinf = R.from_quat(replace_zero_norm_quat(sup)), R.from_quat(replace_zero_norm_quat(inf))
    if order=="inferior_to_superior":
        Rrel = Rinf * Rsup.inv()
    else:
        Rrel = Rsup.inv() * Rinf
    return Rrel.as_quat()


# ---------- Joint construction ----------
def create_dict_with_data(user_dict: dict) -> dict:
    joints = {
        # Ejemplo: se asume definición similar para cada articulación
        "Art. ejemplo": {
            "movimientos": [],
            "marcadores": [],
            "segmentos": {"s_superior": {"inicio":"","fin":""}},
            "centro": promedio_posicion("","", user_dict)
        }
    }
    # Aquí va todo tu diccionario grande 'rangos_movimientos_con_signos'
    return joints


def convertir_a_dict_articulaciones(user_dict: dict, joints_calc: dict, step: int=1) -> dict:
    articul_data = {}
    tiempos = None
    for art, info in joints_calc.items():
        centro = info.get('centro')
        segs = info.get('segmentos', {})
        if 's_superior' in segs and 's_inferior' in segs:
            arr = calcular_cinematic_rel(segs['s_superior'], segs['s_inferior'], user_dict, step)
        else:
            arr = calcular_cinematic_rel(segs.get('s_unico'), None, user_dict, step)
        rot_keys = ['xrot','yrot','zrot','wrot']
        data = {k: arr[:,i] for i,k in enumerate(rot_keys)}
        centro_sub = {k: np.array(centro[k])[::step] for k in ['x','y','z']}
        combined = {**centro_sub, **data}
        articul_data[art] = combined
        if tiempos is None:
            n = combined['x'].shape[0]
            tiempos = {'frames': np.array(user_dict['info']['frames'])[::step][:n],
                       'ms': np.array(user_dict['info']['ms'])[::step][:n]}
    return {'data': articul_data, 'info': tiempos}


# ---------- Rangos de movimiento ----------
rangos_movimientos_con_signos = {
    # copia completa de tu diccionario con rangos de cada articulación
}