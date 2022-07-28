import logging
import comet_ml


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
detail_formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")


def setLogFile(logger: logging.Logger, path: str):
    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(detail_formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


def getLogger(name):
    log = logging.getLogger(name)
    return log


class CommetLogHandler(logging.StreamHandler):
    on_same_line = False

    def __init__(self, exp: comet_ml.Experiment):
        logging.StreamHandler.__init__(self)
        self.exp = exp

    def emit(self, record):
        self.exp.log_text(self.format(record))


def comet_logger(name: str, exp: comet_ml.experiment):
    logger = getLogger(name)
    "%(levelname)s:%(name)s:%(message)s"
    new_handler = CommetLogHandler(exp)
    new_handler.setFormatter(detail_formatter)
    logger.addHandler(new_handler)
    return logger
