import  json, traceback
from app.config import modellogger
from app.utils.model_util import prefill_logmsg
from app.utils.model_exception import *
from jsonschema import validate


def schemaValidated(rawData, ml_schema, sealId, userId,
    requestId="999999999"):
    try:
        schemaData = {}
        schemaData['features'] = rawData
        logMsg = prefill_logmsg(sealId, userId, requestId)
        logMsg['log_subtype'] = 'schemaValidation'
        checkResult = validate(schemaData, ml_schema)
        if checkResult is None:
            logMsg['msg'] = 'Finished checking input data format.'
        modellogger.debug(json.dumps(logMsg))
        return True
    except Exception as e:
        errMsg = traceback.format_exc()
        raise ModelInputSchemaException(errMsg)