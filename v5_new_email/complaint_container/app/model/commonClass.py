from pydantic import BaseModel, constr
#
# class ModelOutput(BaseModel):
#     MLModel: str


class ModelInfoOutput(BaseModel):
    modelInfo: dict


class ModelValideHeader(BaseModel):
    #userId: constr(max_length=7, regex='^[a-zA-Z][0-9]{2,6}$')
    userId: constr(max_length=7)
    sealId: int
    request_id: int
