import pandas as pd
from fastapi import UploadFile
from ..utils.preprocessing_helpers import (
    load_excel_file,
    create_tuple_names_list,
    creating_marker_dict,
    add_data_to_dict
)

class ProcessingFilesService:
    @staticmethod
    async def parse_marker_data(file: UploadFile):
        """
        Convierte un UploadFile de Excel en el diccionario de marcadores completo.
        """
        contents = await file.read()
        df = pd.read_excel(contents, header=None)
        tuples = create_tuple_names_list(df)
        marker_dict = creating_marker_dict(df, tuples)
        full_dict = add_data_to_dict(df, marker_dict, tuples, start_row=1)
        return full_dict