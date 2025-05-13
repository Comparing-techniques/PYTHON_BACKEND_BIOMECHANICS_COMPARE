from fastapi import UploadFile, File, Form, HTTPException

from pydantic import BaseModel, field_validator, ValidationError, model_validator
from fastapi.exceptions import RequestValidationError
from ..handlers.logger import logger


class RequestModel(BaseModel):
    
    base_excel_file: UploadFile
    excel_file_compare: UploadFile
    joint_id: int
    

    @model_validator(mode="before")
    def check_all_present(cls, values:dict):
        missing = [f for f in ("base_excel_file", "excel_file_compare", "joint_id") if f not in values or values[f] is None]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        return values


    @field_validator("base_excel_file", "excel_file_compare")
    def check_file_type(cls, file: UploadFile):
        """
        Validate that the uploaded file is an Excel file.
        """

        logger.info("Biomechanics feedback request received")

        if not file.filename.endswith(('.xlsx', '.xls', '.xlsm')):
            raise ValueError("File must be an Excel file with .xlsx, .xls or .xlsm extension")
        return file
    

    @field_validator("joint_id")
    def check_joint_id(cls, joint_id: int):
        """
        Validate that the joint_id is a positive integer.
        """
        if joint_id <= 0:
            raise ValueError("Joint ID must be a positive integer")
        return joint_id
    
    
    @classmethod
    def validate(cls, 
                base_excel_file: UploadFile = File(..., description="Reference user excel file"), 
                excel_file_compare: UploadFile = File(..., description="User excel file to compare"), 
                joint_id: int = Form(..., description="Joint ID to synchronize in time the movements")) -> "RequestModel":
        """
        Validate the request data.
        """
        try:
            return cls(base_excel_file=base_excel_file, excel_file_compare=excel_file_compare, joint_id=joint_id)
        except ValidationError as e:
            raw = e.errors()
            cleaned = []
            for err in raw:
                # eliminamos el objeto UploadFile
                err.pop("input", None)
                # si hay ctx con excepción, la convertimos a string
                if "ctx" in err and "error" in err["ctx"]:
                    err["ctx"]["error"] = str(err["ctx"]["error"])
                cleaned.append(err)
            # reenviamos como RequestValidationError para que lo capte tu handler
            raise RequestValidationError(cleaned)