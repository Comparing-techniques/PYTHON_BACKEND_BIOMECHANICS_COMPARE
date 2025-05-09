from fastapi import UploadFile
import numpy as np
import pandas as pd
from collections import Counter

from fastapi import UploadFile
import pandas as pd
from io import BytesIO

from fastapi import UploadFile
import pandas as pd
from io import BytesIO


async def load_excel_file(file: UploadFile) -> pd.DataFrame:
    """
    Carga un archivo Excel en un DataFrame desde un UploadFile.

    Parámetros:
    - file (UploadFile): Archivo .xlsx.

    Retorna:
    - pd.DataFrame: DataFrame cargado si tiene éxito.
    - None: Si ocurre un error.
    """
    try:
        # Leer contenido y envolverlo en BytesIO
        content = await file.read()
        df = pd.read_excel(BytesIO(content), header=None)
        print(f"Archivo cargado correctamente: {file.filename}")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{file.filename}' no fue encontrado. (load_excel_file)")
    except ValueError as ve:
        print(f"Error de valor: {ve} (load_excel_file)")
    except Exception as e:
        print(f"Error inesperado: {e} (load_excel_file)")
    return None


def extract_numeric_list(df: pd.DataFrame) -> list[float]:
    """
    Busca la fila con 'Time' en la segunda columna, extrae datos numéricos hacia abajo
    (excluyendo 'Time') hasta la primera celda vacía, y devuelve una lista de floats.
    """
    try:
        # Buscar la fila que contiene 'Time' en la segunda columna (índice 1)
        row_index = df[df.iloc[:, 1].astype(str).str.contains('Time', na=False)].index

        if row_index.empty:
            raise ValueError("La celda 'Time' no fue encontrada en la segunda columna.")

        # Obtener el índice de la fila donde está 'Time'
        start_row = row_index[0]

        # Extraer datos hacia abajo desde la fila siguiente a 'Time'
        time_list = []
        for item in df.iloc[start_row + 1:, 1]:
            if pd.isna(item) or item == '':
                break  # Detener si encuentra celda vacía
            try:
                # Reemplazar coma por punto si existe
                item = str(item).replace(',', '.').strip()

                # Si el valor tiene más de 6 dígitos antes del punto, mover el punto decimal 6 posiciones
                if item.isdigit() and len(item) > 6:
                    # Mover la coma 6 lugares a la izquierda (dividir entre 1,000,000) y redondear a tres decimales
                    item = round(int(item) / 1000000, 4)

                # Convertir a float
                num = float(item)

                time_list.append(num)  # Agregar a la lista
            except ValueError:
                print(f"Advertencia: '{item}' no es un número válido y se omitirá.")

        return time_list

    except ValueError as ve:
        print(f"Error (extract_numeric_list): {ve}")
    except Exception as e:
        print(f"Error inesperado (extract_numeric_list): {e}")
    return []


def create_tuple_names_list(df:pd.DataFrame) -> list[tuple]:
    """Busca la fila con 'Name' en la segunda columna, extrae datos a la derecha y devuelve una lista de tuplas con los conteos."""
    try:
        # Buscar la fila que contiene 'Name' en la columna 2 (índice 1)
        row_index = df[df.iloc[:, 1] == 'Name'].index
       
        if row_index.empty:
            raise ValueError("La celda 'Name' no fue encontrada en la columna 2.")

        # Tomar el índice de la fila encontrada
        row = row_index[0]
        # Extraer datos a la derecha de la columna que contiene 'Name'
        data_list = [
            item.split(":", 1)[1].strip() if isinstance(item, str) and ":" in item else item
            for item in df.iloc[row, 2:].tolist()
        ]
        # Contar las ocurrencias y devolver como lista de tuplas
        return list(Counter(data_list).items())

    except ValueError as ve:
        print(f"Error de valor (create_tuple_names_list): {ve}")
    except IndexError as ie:
        print(f"Error de índice (create_tuple_names_list): {ie}")
    except Exception as e:
        print(f"Error inesperado (create_tuple_names_list): {e}")
    return []


def creating_marker_dict(df: pd.DataFrame, list_tuples) -> dict:
    """
    Crea un diccionario jerárquico con datos de marcadores.

    - "data" contiene un diccionario por marcador, cada uno con claves: x, y, z, xrot, yrot, zrot, wrot, isquat
    - "info" contiene los valores de tiempo (ms) y frame

    Args:
        df (pd.DataFrame): DataFrame de entrada con datos crudos.
        list_tuples (list): Lista de tuplas (nombre_marcador, num_valores).

    Returns:
        dict: Diccionario con estructura organizada.
    """
    try:
        # Extraer lista de milisegundos
        time_list = extract_numeric_list(df.copy())
        if not time_list:
            raise ValueError("La lista de tiempos está vacía.")
        
        num_frames = len(time_list)
        frame_numbers = list(range(1, num_frames + 1))

        # Inicializar estructura
        output = {
            "data": {},
            "info": {
                "ms": time_list,
                "frames": frame_numbers
            }
        }
        
        for name, num_data in list_tuples:
            output["data"][name] = {
                "x": None,
                "y": None,
                "z": None,
                "xrot": None,
                "yrot": None,
                "zrot": None,
                "isQuat": num_data == 7
            }
            if num_data == 7:
                output["data"][name]["wrot"] = np.array
        return output

    except Exception as e:
        print(f"❌ Error: {e}")
        return {}

def add_data_to_dict(df: pd.DataFrame, data_dict: dict, list_tuples: list, start_row: int) -> dict:
    """
    Extrae datos de rotación y posición de 'df' y los agrega al diccionario con estructura organizada.

    Args:
        df (pd.DataFrame): DataFrame con los datos crudos de entrada.
        data_dict (dict): Diccionario generado por 'creating_marker_dict'.
        list_tuples (list): Lista de tuplas con (nombre_marcador, num_valores). Si num_valores == 7, es cuaternión.
        start_row (int): Columna de inicio donde comienzan los datos de cada frame.

    Returns:
        dict: Diccionario actualizado con los valores de posición y rotación.
    """
    try:
        # Buscar la fila que contiene 'Rotation' o 'Position' en la tercera columna (índice 2)
        row_index = df[df.iloc[:, 2].astype(str).str.contains('Rotation|Position', na=False)].index

        if row_index.empty:
            raise ValueError("No se encontró la celda 'Rotation | Position' en la tercera columna.")
        contador = "inicio"
        # Índices de filas clave
        name_marker_row = row_index[0] - 2
        tipe_coord_row = row_index[0] + 1
        position_row = row_index[0]
        
        # Mapas de coordenadas para rotación y posición
        rotation_map = {"X": "xrot", "Y": "yrot", "Z": "zrot", "W": "wrot"}
        position_map = {"X": "x", "Y": "y", "Z": "z"}
       
        num_frames = len(data_dict["info"]["ms"])
        # Iterar por columnas de datos
        for col in df.columns[start_row:]:
            col_idx = df.columns.get_loc(col)
            marker_name = str(df.iloc[name_marker_row, col_idx]).split(':')[-1].strip()
            tipe_coord = str(df.iloc[tipe_coord_row, col_idx]).strip()
            position = str(df.iloc[position_row, col_idx]).strip()
            contador = marker_name
            # Datos numéricos desde la fila siguiente a tipe_coord_row
            data = pd.to_numeric(df.iloc[tipe_coord_row + 1:, col], errors='coerce').to_numpy()
            if len(data) != num_frames:
                continue  # Saltar si el número de frames no coincide


            # Asignar los datos según tipo y coordenada
            if position == "Rotation" and tipe_coord in rotation_map:
                data_dict["data"][marker_name][rotation_map[tipe_coord]] = np.array(data)

            elif position == "Position" and tipe_coord in position_map:
                data_dict["data"][marker_name][position_map[tipe_coord]] = np.array(data)
        return data_dict

    except ValueError as ve:
        print(f"❌ Error de valor: {ve}")
    except Exception as e:
        print(contador)
        print(f"⚠️ Error inesperado (add_data_to_dict): {e}")

    return data_dict  # Devuelve el diccionario sin cambios en caso de error


def add_data_to_dict(df: pd.DataFrame, data_dict: dict, list_tuples: list, start_row: int) -> dict:
    """
    Extrae datos de rotación y posición de 'df' y los agrega al diccionario con estructura organizada.

    Args:
        df (pd.DataFrame): DataFrame con los datos crudos de entrada.
        data_dict (dict): Diccionario generado por 'creating_marker_dict'.
        list_tuples (list): Lista de tuplas con (nombre_marcador, num_valores). Si num_valores == 7, es cuaternión.
        start_row (int): Columna de inicio donde comienzan los datos de cada frame.

    Returns:
        dict: Diccionario actualizado con los valores de posición y rotación.
    """
    try:
        # Buscar la fila que contiene 'Rotation' o 'Position' en la tercera columna (índice 2)
        row_index = df[df.iloc[:, 2].astype(str).str.contains('Rotation|Position', na=False)].index

        if row_index.empty:
            raise ValueError("No se encontró la celda 'Rotation | Position' en la tercera columna.")
        contador = "inicio"
        # Índices de filas clave
        name_marker_row = row_index[0] - 2
        tipe_coord_row = row_index[0] + 1
        position_row = row_index[0]
        
        # Mapas de coordenadas para rotación y posición
        rotation_map = {"X": "xrot", "Y": "yrot", "Z": "zrot", "W": "wrot"}
        position_map = {"X": "x", "Y": "y", "Z": "z"}
       
        num_frames = len(data_dict["info"]["ms"])
        # Iterar por columnas de datos
        for col in df.columns[start_row:]:
            col_idx = df.columns.get_loc(col)
            marker_name = str(df.iloc[name_marker_row, col_idx]).split(':')[-1].strip()
            tipe_coord = str(df.iloc[tipe_coord_row, col_idx]).strip()
            position = str(df.iloc[position_row, col_idx]).strip()
            contador = marker_name
            # Datos numéricos desde la fila siguiente a tipe_coord_row
            data = pd.to_numeric(df.iloc[tipe_coord_row + 1:, col], errors='coerce').to_numpy()
            if len(data) != num_frames:
                continue  # Saltar si el número de frames no coincide


            # Asignar los datos según tipo y coordenada
            if position == "Rotation" and tipe_coord in rotation_map:
                data_dict["data"][marker_name][rotation_map[tipe_coord]] = np.array(data)

            elif position == "Position" and tipe_coord in position_map:
                data_dict["data"][marker_name][position_map[tipe_coord]] = np.array(data)
        return data_dict

    except ValueError as ve:
        print(f"❌ Error de valor: {ve}")
    except Exception as e:  
        print(f"⚠️ Error inesperado (add_data_to_dict): {e}")

    return data_dict  # Devuelve el diccionario sin cambios en caso de error