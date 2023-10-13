import json
import os, re, traceback
from typing import Union, Optional, Any, List

from fastapi import APIRouter, Header, HTTPException
from app.config import modellogger
from app.model import myModels
from app.model.complaint_api_models import ComplaintAPIRequest
from app.process.data_preprocess import preprocess
from app.utils.model_util import prefill_logmsg
from app.utils.model_exception import *
from app.utils.app_utils import get_nltk_data_dir
from time import time
from app import mlesm_logger

complaint_router = APIRouter(
    tags=["ML Serving Complaint Model Endpoints"],
    responses={200: {"description": "Success"},
               400: {"description": "Fail at format"},
               500: {"description": "System Failure"}
               }
)

mlesm_logger.info("model_source : " + myModels.model_source)
modellogger.info("Running model from " + myModels.model_source)

seal_match = re.compile(r'^[0-9]{1,10}$')
user_match = re.compile(r'^[a-zA-Z]{1}[0-9]{3,10}$')
requestid_match = re.compile(r'^[0-9a-zA-Z-]{1,60}$')

def on_startup():
    nltk_data_dir = get_nltk_data_dir()
    mlesm_logger.info(f"on_startup : setup nltk_data {nltk_data_dir}")

def postprocess(y_predicts, best_threshold):
    raw_output = [1 if y_predict > best_threshold else 0 for y_predict in y_predicts]
    pred_1=y_predicts
    pred_0=1-y_predicts

    translate = { 1: 'Complaint', 0: 'Non-Complaint'}
    translated_results = [{"pred_response": translate[i],"pred_1":j,"pred_0":k} for i,j,k in zip(raw_output,pred_1,pred_0)]
    # translated_results.update()
    return translated_results

@complaint_router.post(path="")
async def complaint_handler(complaint_api_requests: List[ComplaintAPIRequest],
                              optionalHeader: Optional[Union[str, None]] = Header(default=None),
                              X_JPMC_SEAL_ID: Optional[str] = Header(default=None),
                              X_JPMC_USER_ID: Optional[str] = Header(default=None),
                              request_id: Optional[str] = Header(default=None)) -> Any:
    try:

        seal_chk = seal_match.match(str(X_JPMC_SEAL_ID))
        user_chk = user_match.match(str(X_JPMC_USER_ID))
        requestId_chk = requestid_match.match(str(request_id))
        logMsg = {}
        logMsg = prefill_logmsg(X_JPMC_SEAL_ID, X_JPMC_USER_ID, request_id)
        # use Any can make it look better , but will consume more CPU
        if seal_chk is None:
            raise ModelHeaderSealIdFormatException(X_JPMC_SEAL_ID)
        if user_chk is None:
            raise ModelHeaderUserIdFormatException(X_JPMC_USER_ID)
        if requestId_chk is None:
            raise ModelHeaderReqIdFormatException(request_id)

        input_raw = []
        for complaint_api_request in complaint_api_requests:
            complaint_dict = json.loads(complaint_api_request.json())
            input_raw.append(complaint_dict)

        sealId = X_JPMC_SEAL_ID
        userId = X_JPMC_USER_ID
        startt = time()

        startt_preprocess = time()
        processed_data = preprocess(input_raw, myModels)

        logMsg['preprocess_responsetime_ms'] = (time() - startt_preprocess) * 1000

        booster = myModels.predict_model
        y_predicts = booster.predict(processed_data)
        best_threshold = myModels.best_threshold
        # raw_output = [1 if y_predict > best_threshold else 0 for y_predict in y_predicts]
        # target = [1 if inp["is_complaint"] == "Y" else 0 for inp in input_raw]
        final_predict = postprocess(y_predicts, best_threshold)


        columns = processed_data.columns.values
        responses = []

        for i in range(len(complaint_api_requests)):
            response = {}
            response['modelResponse'] = final_predict[i]

            row = processed_data.values[i]

            features = {}
            features.update({'snapshot_id': complaint_api_requests[i].snapshot_id,
                             'thread_id': complaint_api_requests[i].thread_id,
                             'gcid': complaint_api_requests[i].gcid,
                             'time_variable': complaint_api_requests[i].time,
                             # 'target_variable': target[i],
                             'pred_1': y_predicts[i],
                             'pred_0': 1-y_predicts[i],
                             # 'classifier_threshold': best_threshold
                             })
            for j in range(len(columns)):
                col_name = columns[j]
                data = {col_name:row[j]}
                features.update(data)
            response['modelRequest'] = features
            responses.append(response)


        logMsg['prediction_responsetime_ms'] = (time() - startt) * 1000
        logMsg['response_status_code'] = '200'
        logMsg['input'] = input_raw
        logMsg['output'] = final_predict
        mlesm_logger.info(json.dumps(logMsg))

        return responses

    except (ModelHeaderSealIdFormatException,
           ModelHeaderUserIdFormatException,
           ModelHeaderReqIdFormatException) as he:
        logMsg['response_status_code'] = he.status_code
        logMsg['errmsg'] = he.message
        mlesm_logger.error(json.dumps(logMsg))
        raise HTTPException(status_code=he.status_code, detail=he.message)
    except (ModelPreProcessException,
          ModelSchemaException,
          ModelInputSchemaException) as pe:
        logMsg['response_status_code'] = pe.status_code
        logMsg['errmsg'] = pe.message
        mlesm_logger.error(json.dumps(logMsg))
        raise HTTPException(status_code=pe.status_code, detail=pe.message)
    except Exception as e:
        logMsg = {}
        logMsg = prefill_logmsg(X_JPMC_SEAL_ID, X_JPMC_USER_ID, request_id)
        errMsg = traceback.format_exc()
        logMsg['response_status_code'] = 500
        logMsg['errmsg'] = errMsg
        mlesm_logger.error(json.dumps(logMsg))
        raise HTTPException(status_code=500, detail=errMsg)


@complaint_router.get(path="")
def describe_model() -> Any:
    modelInfo = {}
    modelInfo["MLModel"] = str(myModels.predict_model)
    modelInfo["MODEL_NAME"] = os.environ.get('MODEL_NAME')
    modelInfo["MODEL_VERSION"] = myModels.model_version
    modelInfo["MODEL_PATH"] = myModels.model_source
    if os.environ.get("MODEL_SCHEMAFILE") is not None:
        modelInfo["MODEL_SCHEMA"] =  os.path.join( myModels.model_source,
                                            os.environ.get("MODEL_SCHEMAFILE"))
    ModelInfoOutput = {}
    ModelInfoOutput['modelInfo'] = modelInfo
    logMsg = {}
    logMsg['log_type'] = 'ModelInfo'
    logMsg['log_subtype'] = 'Query'
    logMsg['msg'] = 'The serving model info {}'.format(modelInfo)
    modellogger.info(json.dumps(logMsg))
    return ModelInfoOutput