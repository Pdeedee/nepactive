import logging
import os
    
ROOT_PATH = __path__[0]
NAME = "nepactive"
SHORT_CMD = "nepactive"
dlog = logging.getLogger(__name__)
dlog.setLevel(logging.INFO)
dlogf = logging.FileHandler(os.getcwd() + os.sep + SHORT_CMD + ".log", delay=True)
dlogf_formatter = logging.Formatter("%(asctime)s - %(levelname)s : %(message)s")
# dlogf_formatter=logging.Formatter('%(asctime)s - %(name)s - [%(filename)s:%(funcName)s - %(lineno)d ] - %(levelname)s \n %(message)s')
dlogf.setFormatter(dlogf_formatter)
dlog.addHandler(dlogf)
logging.basicConfig(level=logging.WARNING)