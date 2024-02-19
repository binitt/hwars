import logging
import logging.handlers
import sys

def logging_init_stdout():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def logging_init_file():
    root_logger= logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
            'logs/trace.log', maxBytes=50 * 1024 * 1024, 
            backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
            '%(levelname)s %(threadName)s %(asctime)s %(filename)s:%(lineno)d %(funcName)s %(message)s'))
    root_logger.addHandler(handler)
    root_logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Logger initialized")