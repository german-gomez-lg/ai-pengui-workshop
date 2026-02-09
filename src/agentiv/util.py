from os import getenv
from dotenv import dotenv_values
import logging


def is_running_locally() -> bool:
    """Check if is running locally or not"""
    databricks_app_name = getenv("DATABRICKS_APP_NAME", "")
    if not databricks_app_name:
        return True
    return False

def from_env_or_dotenv(env_var_name:str, default:str) -> str:
    running_locally = is_running_locally()
    values = dotenv_values('.env') if running_locally else {}

    if running_locally:
        return values[env_var_name] if env_var_name in values else default
    
    return getenv(env_var_name, default)


def _get_log_level()->str:
    """Get the log level from the env variable LOG_LEVEL"""
    name_to_level = {
        'CRITICAL': logging.CRITICAL,
        'FATAL': logging.FATAL,
        'ERROR': logging.ERROR,
        'WARN': logging.WARNING,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG
    }

    log_level_str = getenv('LOG_LEVEL') 
    
    if log_level_str:
        log_level_str = log_level_str.upper().strip()

    if log_level_str not in name_to_level:
        return logging.DEBUG
    
    return name_to_level[log_level_str]

def get_named_logger(name: str) -> logging.Logger:
    """Factory method to create loggers adjusted to our preferences"""
    #create named logger
    logger = logging.getLogger(name)

    #lets not risk duplicating handlers
    if not logger.handlers:
        log_level = _get_log_level()
        logger.setLevel(log_level)

        # Create handler and formatter
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        #formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [%(funcName)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False # I dont want to propagate this to root and end up duplicating logs

    return logger