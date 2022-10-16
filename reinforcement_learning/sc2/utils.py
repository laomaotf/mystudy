import logging
import sys,os
from logging import handlers


###########################################
#LOGGING
log_level_str2var = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR
}
def get_logger(filepath, levelstr="info"):
    os.makedirs(os.path.dirname(filepath),exist_ok=True)
    log = logging.getLogger(filepath)
    level = log_level_str2var[levelstr]
    log.setLevel(level)
    fmt = logging.Formatter('%(filename)s-%(lineno)d %(levelname)s:%(message)s')
    screen = logging.StreamHandler(sys.stdout)
    screen.setLevel(level)
    screen.setFormatter(fmt)
    
    file = handlers.RotatingFileHandler(filename=filepath,
                                      maxBytes=1024*1024*100,backupCount=5)
    file.setLevel(level)
    file.setFormatter(fmt)
    log.addHandler(screen)
    log.addHandler(file)
    return log

logger = get_logger(os.path.join(os.path.dirname(__file__),"pysc2.log"))