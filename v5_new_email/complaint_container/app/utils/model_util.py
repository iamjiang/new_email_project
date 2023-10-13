from app.config import modellogger, model_name
from app.model import myModels

def prefill_logmsg(sealId, userId, requestId):
    logMsg = {}
    logMsg['log_type'] = 'Model'
    logMsg['log_subtype'] = 'Prediction'
    logMsg['sealId'] = sealId
    logMsg['userId'] = userId
    logMsg['requestId'] = requestId
    logMsg['model_name'] = model_name
    logMsg['model_version'] = myModels.model_version
    logMsg['model_source'] = myModels.model_source
    return logMsg

