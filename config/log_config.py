from pathlib import Path

from cscs_tools.logger.services.logger_service import LoggerService

from config.config import LOG_LEVEL, LOG_FORMAT, LOG_TO_STDOUT

LOG_DIRECTORY = str(Path(__file__).resolve().parent.parent / "log")

error_logger_service = LoggerService(
    log_directory=LOG_DIRECTORY,
    log_file="error.log",
    level=LOG_LEVEL,
    log_format=LOG_FORMAT,
    to_stdout=LOG_TO_STDOUT
)
access_logger_service = LoggerService(
    log_directory=LOG_DIRECTORY,
    log_file="access.log",
    level=LOG_LEVEL,
    log_format=LOG_FORMAT,
    to_stdout=LOG_TO_STDOUT,
    days=2
)