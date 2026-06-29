import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def get_logger() -> logging.Logger:
    """App logger that writes to logs/api.log (rotating) and the console."""
    logger = logging.getLogger("pi-api")
    if logger.handlers:  # already configured
        return logger

    LOG_DIR.mkdir(exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")

    file_handler = RotatingFileHandler(LOG_DIR / "api.log", maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler()
    console.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger
