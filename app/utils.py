import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename='sync.log',
    filemode='a' 
)

logger = logging.getLogger(__name__)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(console_handler)


def log_info(message):
    """
    Log informational messages.
    
    Args:
        message (str): The message to log as INFO level.
    """
    logger.info(message)


def log_error(message):
    """
    Log error messages.
    
    Args:
        message (str): The message to log as ERROR level.
    """
    logger.error(message)
