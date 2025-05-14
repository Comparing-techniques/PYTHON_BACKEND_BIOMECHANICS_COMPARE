# Mensajes de error para processing_files.py
FILE_NOT_FOUND_ERROR = "El archivo '{filename}' no fue encontrado."
VALUE_ERROR_MSG = "Error de valor: {error}"
UNEXPECTED_ERROR_MSG = "Error inesperado: {error}"

EXTRACT_NUMERIC_LIST_TIME_NOT_FOUND = ("La celda 'Time' no fue encontrada en la segunda columna. (extract_numeric_list)")
EXTRACT_NUMERIC_LIST_VALUE_WARNING = ("Advertencia: '{item}' no es un número válido y se omitirá. (extract_numeric_list)")
EXTRACT_NUMERIC_LIST_ERROR = ("Error (extract_numeric_list): {error}")
EXTRACT_NUMERIC_LIST_UNEXPECTED = ("Error inesperado (extract_numeric_list): {error}")

CREATE_TUPLE_NAMES_NAME_NOT_FOUND = ("La celda 'Name' no fue encontrada en la columna 2. (create_tuple_names_list)")
CREATE_TUPLE_NAMES_VALUE_ERROR = ("Error de valor (create_tuple_names_list): {error}")
CREATE_TUPLE_NAMES_INDEX_ERROR = ("Error de índice (create_tuple_names_list): {error}")
CREATE_TUPLE_NAMES_UNEXPECTED_ERROR = ("Error inesperado (create_tuple_names_list): {error}")

TIME_LIST_NOT_EMPTY = ("La lista de tiempos está vacía.")
ERROR_IN_ROTATION = ("No se encontró la celda 'Rotation | Position' en la tercera columna.")
ADD_DATA_TO_DICT_VALUE_ERROR = ("Error de valor (add_data_to_dict): {error}")
ADD_DATA_TO_DICT_UNEXPECTED_ERROR = ("⚠️ Error inesperado (add_data_to_dict): {error}")