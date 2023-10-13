import os, logging
from app import ROOT_DIR
import yaml


class BaseConfig():
    API_PREFIX = '/api'
    MLMODEL_PATH = os.path.join(ROOT_DIR,  'app', 'model')
    default_logging_level = 'logging.DEBUG'
    LOG_LEVEL = os.getenv('LOG_LEVEL', default_logging_level)
    APP_LOGS = '/app/logs/serving.log'
    MODEL_SOURCE = os.getenv('MDLC_SOURCE',None)
    PROJECT_NAME = os.getenv('PROJECT_NAME','NEXLA Integration TRN')
    MODEL_NAME = os.getenv('MODEL_NAME','cb-complaints-tfidf')
    MODEL_VERSION = os.getenv('MODEL_VERSION',
                              'LOCAL')


class localConfig(BaseConfig):
    MLMODEL_PATH = os.path.join(ROOT_DIR,  'app', 'model')
    APP_LOGS = 'I:\\tmp\\logs\\serving.log'
    MDLC_PATH = os.path.join(ROOT_DIR,'mdlc')
    UPLOAD_DIR = os.path.join(ROOT_DIR,'upload')


class devConfig(BaseConfig):
    FLASK_ENV = 'development'
    LOG_LEVEL = os.getenv('LOG_LEVEL', logging.INFO)


class prodConfig(BaseConfig):
    FLASK_ENV = 'production'
    LOG_LEVEL = os.getenv('LOG_LEVEL', logging.INFO)


class testConfig(BaseConfig):
    FLASK_ENV = 'test'
    LOG_LEVEL = 'logging.INFO'