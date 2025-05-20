import numpy as np
import pandas as pd


def compute_kinematics(user_data: dict) -> dict:
    """
    Given user_data with 'info': {'ms': [...]}
    and 'data': {art: {'x', 'y', 'z'} arrays}, compute per-frame:
      - velocity components vx, vy, vz
      - acceleration components ax, ay, az
      - jerk (derivative of acceleration) components jx, jy, jz
      - velocity magnitude, acceleration magnitude, jerk magnitude
      - sign of velocity per axis

    Returns:
      kinematics: dict of articulations to dict with keys:
        'ms': array of times for valid frames
        'vel', 'acc', 'jerk': arrays of magnitudes
        'signs': dict of 'vx', 'vy', 'vz' sign arrays
    """
    ms = np.array(user_data['info']['ms'], dtype=float)
    dt = np.diff(ms) / 1000.0  # seconds
    kinematics = {}

    for art, d in user_data['data'].items():
        x, y, z = d['x'], d['y'], d['z']
        vx = np.diff(x) / dt
        vy = np.diff(y) / dt
        vz = np.diff(z) / dt
        ax = np.diff(vx) / dt[:-1]
        ay = np.diff(vy) / dt[:-1]
        az = np.diff(vz) / dt[:-1]
        jx = np.diff(ax) / dt[:-2]
        jy = np.diff(ay) / dt[:-2]
        jz = np.diff(az) / dt[:-2]

        vel_mag = np.sqrt(vx**2 + vy**2 + vz**2)
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        jerk_mag = np.sqrt(jx**2 + jy**2 + jz**2)

        sign_vx = np.sign(vx)
        sign_vy = np.sign(vy)
        sign_vz = np.sign(vz)

        kinematics[art] = {
            'ms_vel': ms[1:],
            'ms_acc': ms[2:],
            'ms_jerk': ms[3:],
            'vel_mag': vel_mag,
            'acc_mag': acc_mag,
            'jerk_mag': jerk_mag,
            'sign_vx': sign_vx,
            'sign_vy': sign_vy,
            'sign_vz': sign_vz
        }
    return kinematics


def sliding_window_indices(ms_array: np.ndarray, window_ms: int = 10) -> list:
    """
    Generate a list of (start_idx, end_idx) for sliding non-overlapping windows of window_ms
    based on timestamps ms_array.
    ms_array: increasing array of times in ms.
    window_ms: width of each window in ms.
    """
    indices = []
    start_idx = 0
    n = len(ms_array)
    while start_idx < n:
        t0 = ms_array[start_idx]
        end_idx = np.searchsorted(ms_array, t0 + window_ms, side='right')
        if end_idx <= start_idx:
            break
        indices.append((start_idx, end_idx))
        start_idx = end_idx
    return indices


def compute_window_metrics(kin_ref: dict, kin_cmp: dict, window_ms: int = 10) -> dict:
    """
    Compare kinematics of reference and comparison user for each articulation.
    Returns dict of articulation -> DataFrame with columns:
      inicio_ms, fin_ms, n_frames,
      vel_media, acel_media, cambios_direccion, jerk_medio, jerk_std,
      vel_media_diff_pct, acel_media_diff_pct, jerk_medio_diff_pct, cambios_direccion_diff
    """
    results = {}

    for art, ref in kin_ref.items():
        if art not in kin_cmp:
            continue

        cmp = kin_cmp[art]
        ms = ref['ms_vel']
        windows = sliding_window_indices(ms, window_ms)
        rows = []

        for i0, i1 in windows:
            v_ref = ref['vel_mag'][i0:i1]
            v_cmp = cmp['vel_mag'][i0:i1]
            a_ref = ref['acc_mag'][max(i0 - 1, 0):i1 - 1]
            a_cmp = cmp['acc_mag'][max(i0 - 1, 0):i1 - 1]
            j_ref = ref['jerk_mag'][max(i0 - 2, 0):i1 - 2]
            j_cmp = cmp['jerk_mag'][max(i0 - 2, 0):i1 - 2]

            vel_mean_ref = np.mean(v_ref)
            vel_mean_cmp = np.mean(v_cmp)
            acc_mean_ref = np.mean(a_ref) if len(a_ref) > 0 else np.nan
            acc_mean_cmp = np.mean(a_cmp) if len(a_cmp) > 0 else np.nan
            jerk_mean_ref = np.mean(j_ref) if len(j_ref) > 0 else np.nan
            jerk_mean_cmp = np.mean(j_cmp) if len(j_cmp) > 0 else np.nan
            jerk_std_cmp = np.std(j_cmp) if len(j_cmp) > 0 else np.nan

            def count_sign_changes(sign_arr):
                return int(np.sum(sign_arr[:-1] * sign_arr[1:] < 0))

            cd_ref = (
                count_sign_changes(ref['sign_vx'][i0:i1]) +
                count_sign_changes(ref['sign_vy'][i0:i1]) +
                count_sign_changes(ref['sign_vz'][i0:i1])
            )
            cd_cmp = (
                count_sign_changes(cmp['sign_vx'][i0:i1]) +
                count_sign_changes(cmp['sign_vy'][i0:i1]) +
                count_sign_changes(cmp['sign_vz'][i0:i1])
            )

            eps = 1e-8
            vel_diff_pct = 100 * (vel_mean_cmp - vel_mean_ref) / (vel_mean_ref + eps)
            acc_diff_pct = 100 * (acc_mean_cmp - acc_mean_ref) / (acc_mean_ref + eps)
            jerk_diff_pct = 100 * (jerk_mean_cmp - jerk_mean_ref) / (jerk_mean_ref + eps)
            cd_diff = cd_cmp - cd_ref

            rows.append({
                'inicio_ms': ms[i0],
                'fin_ms': ms[i1 - 1],
                'n_frames': i1 - i0,
                'vel_media': vel_mean_cmp,
                'acel_media': acc_mean_cmp,
                'cambios_direccion': cd_cmp,
                'jerk_medio': jerk_mean_cmp,
                'jerk_std': jerk_std_cmp,
                'vel_media_diff_pct': vel_diff_pct,
                'acel_media_diff_pct': acc_diff_pct,
                'jerk_medio_diff_pct': jerk_diff_pct,
                'cambios_direccion_diff': cd_diff
            })

        results[art] = pd.DataFrame(rows)

    return results


def feedback_window_metrics_acel_vel(acel_vel_dict: dict) -> dict:
    """
    Dado un diccionario de DataFrames por articulación, agrega una columna de retroalimentación a cada DataFrame.
    Retorna un diccionario de articulación -> DataFrame con la columna de retroalimentación agregada.
    """
    feedback_acel_vel_dict = {}
    for joint in acel_vel_dict:
        df_feedback = generate_window_feedback(acel_vel_dict[joint])
        for idx, texto in df_feedback['feedback'].items():
            if feedback_acel_vel_dict.get(joint) is None:
                feedback_acel_vel_dict[joint] = " - " + texto
            else:
                feedback_acel_vel_dict[joint] += " - " + texto
    return feedback_acel_vel_dict

def pipeline_compare_users(user_ref: dict, user_cmp: dict, window_ms: int = 10) -> dict:
    """
    Full pipeline: compute kinematics for both users, then windowed comparison metrics.
    Returns dict of DataFrames per articulation.
    """
    kin_ref = compute_kinematics(user_ref)
    kin_cmp = compute_kinematics(user_cmp)
    return compute_window_metrics(kin_ref, kin_cmp, window_ms)


def generate_window_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade una columna 'feedback' al DataFrame con mensajes en palabras para cada ventana.
    Cada mensaje indica el número de frames, tiempo de inicio y fin, y describe:
      - si la velocidad media fue mayor o menor que el modelo
      - si la aceleración media fue mayor o menor que el modelo
      - si el jerk medio indica movimiento brusco
      - si hubo más cambios de dirección que el modelo

        VM: Velocidad media
        AM: Aceleración media
        MB: Movimiento brusco
        MS: Movimiento suave
    """
    feedbacks = []

    for _, row in df.iterrows():
        inicio = row['inicio_ms']
        fin = row['fin_ms']
        n = int(row['n_frames'])

        v_diff = row['vel_media_diff_pct']
        if v_diff > 5:
            v_msg = f"VM {v_diff:.1f}% mayor"
        elif v_diff < -5:
            v_msg = f"VM {abs(v_diff):.1f}% menor"
        else:
            v_msg = "VM similar"

        a_diff = row['acel_media_diff_pct']
        if a_diff > 10:
            a_msg = f"AM {a_diff:.1f}% mayor"
        elif a_diff < -10:
            a_msg = f"AM {abs(a_diff):.1f}% menor"
        else:
            a_msg = "AM aceptable"

        j_diff = row['jerk_medio_diff_pct']
        if j_diff > 10:
            j_msg = "MB (jerk alto) respecto al modelo"
        else:
            j_msg = "MS"

        # c_diff = row['cambios_direccion_diff']
        # if c_diff > 0:
        #     c_msg = f"{c_diff} cambios de dirección más"
        # elif c_diff < 0:
        #     c_msg = f"{abs(c_diff)} cambios de dirección menos"
        # else:
        #     c_msg = "Igual número de cambios de dirección"

        msg = f"Ventana {inicio:.2f}–{fin:.2f} s " + "; ".join([v_msg, a_msg, j_msg]) + "."
        feedbacks.append(msg)

    df_with_feedback = df.copy()
    df_with_feedback['feedback'] = feedbacks
    return df_with_feedback