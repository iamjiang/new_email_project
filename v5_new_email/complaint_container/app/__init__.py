import os
import logging
from logging.handlers import RotatingFileHandler

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
print(ROOT_DIR)

mlesm_logger = logging.getLogger('serving')
fmt_string = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt=fmt_string)
log_level = os.environ.get('LOG_LEVEL', logging.INFO)
mlesm_logger.setLevel(log_level)
CREATE_LOGFILE = (os.environ.get('CREATE_LOGFILE', False) == 'True')
if CREATE_LOGFILE:
    rotatehandler = RotatingFileHandler(
        filename=os.environ.get('FILEPATH', '/app/logs/serving.log'),
        maxBytes=104857600, backupCount=2)
    rotatehandler.setFormatter(formatter)
    mlesm_logger.addHandler(rotatehandler)
else:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    mlesm_logger.addHandler(ch)
print('Mlesm Logger initialized')