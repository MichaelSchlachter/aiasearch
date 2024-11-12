import logging

PROJECT_NAME = "aiasearch"

def log_initialize():
    logger = logging.getLogger(PROJECT_NAME)
    logger.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    logger_console_handler = logging.StreamHandler()
    logger_console_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_console_handler)

    """
    logger_file_handler = logging.FileHandler(PROJECT_NAME + ".log")
    logger_file_handler.setLevel(logging.INFO)
    logger_file_handler.setFormatter(logger_formatter)
    """
