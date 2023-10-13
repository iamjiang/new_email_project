import sys
import app.config.settings
import logging, os
from logging.handlers import RotatingFileHandler
from app import ROOT_DIR


# create settings object corresponding to specified env
APP_ENV = os.environ.get('APP_ENV', 'Dev').lower()
_current = getattr(sys.modules['app.config.settings'], '{0}Config'.format(APP_ENV))()

# copy attributes to the module for convenience
for atr in [f for f in dir(_current) if not '__' in f]:
    # environment can override anything
    val = os.environ.get(atr, getattr(_current, atr))
    setattr(sys.modules[__name__], atr, val)

def as_dict():
    res = {}
    for atr in [f for f in dir(_current) if not '__' in f]:
        val = getattr(_current, atr)
        res[atr] = val
    return res


def setup_logger(logname, logfile, fmtString, logLvL=logging.INFO):
    logger = logging.getLogger(logname)
    logger.setLevel(logLvL)
    formatter = logging.Formatter(fmt=fmtString)
    CREATE_LOGFILE = ( os.environ.get('CREATE_LOGFILE', False) == 'True')
    if CREATE_LOGFILE:
        rotatehandler = RotatingFileHandler(filename=  logfile,
                                            maxBytes=104857600, backupCount=2)
        rotatehandler.setFormatter(formatter)
        logger.addHandler(rotatehandler)
    else:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.info('Start the {} Logger'.format(logname))
    return logger

model_name = os.environ.get('APP_NAME', 'model')
model_ver = os.environ.get('MODEL_VERSION', '1.1.1')

fmt_string = '[%(asctime)s] p%(process)s %(levelname)s - %(message)s'
default_logdir = ROOT_DIR
log_dir =  os.environ.get('MODEL_LOG_DIR', default_logdir)
if os.environ.get('MODEL_NAME'):
    log_file = os.environ.get('MODEL_NAME') + '.log'
    model_name = os.environ.get('MODEL_NAME')
else:
    log_file = os.environ.get('APP_NAME', 'modelserving.log')
log_level = os.environ.get('LOG_LEVEL', logging.INFO)
modellogger = setup_logger(model_name, os.path.join(log_dir, log_file),
                           fmt_string, log_level)