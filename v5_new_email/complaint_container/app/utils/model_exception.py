class ModelHeaderSealIdFormatException(Exception):
    def __init__(self, value, message=None):
        if message is None:
            message = f'X_JPMC_SEAL_ID: {value} has bad format. It needs to ' \
                    f'be all digits ' \
                    'with length between 1 and 10.'
        self.message = message
        self.status_code = 400
        super().__init__(message)


class ModelHeaderUserIdFormatException(Exception):
    def __init__(self, value, message=None):
        if message is None:
            message = f'X_JPMC_USER_ID: {value} has bad format. It needs to start ' \
                    f'with a character that follows with 3 to 10 digits.'
        self.message = message
        self.status_code = 400
        super().__init__(message)


class ModelHeaderReqIdFormatException(Exception):
    def __init__(self, value, message=None):
        if message is None:
            message = f'request_id: {value} has bad format. It needs to follow the ' \
                    f'standard for uuid , contains digits, char or dash' \
                    f'with the length between 1 and 60.'
        self.message = message
        self.status_code = 400
        super().__init__(message)


class ModelPreProcessException(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Preprocsss abended"
        #less babbling
        self.message = "Preprocsss abended with this err:" + message
        self.status_code = 400
        super().__init__(message)


class ModelInputSchemaException(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Input data failed at Schema Validation"
        #less babbling
        self.message = "Input Data Schema Validation abended with this err:" + \
                       message
        self.status_code = 404
        super().__init__(message)


class ModelSchemaException(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "failed at Schema Validation"
        #less babbling
        self.message = "Schema Validation with this err:" + message
        self.status_code = 404
        super().__init__(message)


class ModelException(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Model Prediction abended with this err : "
        self.message = message
        self.status_code = 500