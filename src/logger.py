import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

number_in_megabytes = 5 * 1024 * 1024  # 5MB
number_of_rotating_files = 3

file_handler = RotatingFileHandler(
    LOG_FILE_PATH,
    maxBytes=number_in_megabytes,
    backupCount=number_of_rotating_files,
)

stream_handler = logging.StreamHandler()

formatter = logging.Formatter(
    "[ %(asctime)s ] - %(levelname)s -  %(name)s:%(lineno)d - %(message)s"
)

file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

if __name__ == "__main__":
    logger.info("Logger initialized successfully.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
