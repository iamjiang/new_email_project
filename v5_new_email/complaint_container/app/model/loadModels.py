import joblib, pickle, os, json
import jsonschema, traceback
from app import config
from app.utils.model_exception import ModelSchemaException

def get_schema(schema_file):
    try:
        with open(schema_file, 'r') as file:
            schema = json.load(file)
            print(jsonschema.Draft4Validator.check_schema(schema))
            return schema
    except Exception as e:
        errMsg = traceback.format_exc()
        raise ModelSchemaException(errMsg)
    
class PrepareModels:
    def __init__(self):
        
        model_source = config.MLMODEL_PATH
        model_version = os.environ.get("MODEL_VERSION",config.MODEL_VERSION)
        
        model_dict =joblib.load(os.path.join(model_source,"lightgbm.pkl"))
        
        self.best_threshold=model_dict["best_threshold"]
        self.predict_model=model_dict["model"]
        self.bow_vectorizer=model_dict["bow_vectorizer"]
        self.model_version = model_version
        self.model_source = model_source            
        if os.environ.get("MODEL_SCHEMAFILE") is not None:
            schema_path = os.path.join(config.MLMODEL_PATH,
                                       os.environ.get("MODEL_SCHEMAFILE"))
            self.ml_schema = get_schema(schema_path)
        
