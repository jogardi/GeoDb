import logging


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
adam_log = logging.getLogger("adam")


def setLogFile(path: str):
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    adam_log.addHandler(fh)
    logging.getLogger("adam").setLevel(logging.DEBUG)


def getLogger(name):
    log = logging.getLogger(name)
    return log
