import logging
import os
from pathlib import Path


class VisibleColorFormatter(logging.Formatter):
    def format(self, record):
        RESET = "\033[0m"

        # Level-name coloring stays the same
        if record.levelname == "INFO":
            level_color = ""
            levelname = ""
        else:
            level_color = "\033[91m"
            levelname = f"{level_color}[{record.levelname}]{RESET}"

        # New: combine full path + line into one linkified segment
        location_color = "\033[94m"  # pick your color
        relpath = os.path.relpath(record.pathname, os.getcwd())
        location = f"{location_color}{relpath}:{record.lineno}{RESET}"

        # If you still want funcName, append it after:
        func_color = "\033[96m"
        funcname = f"{func_color}({record.funcName}){RESET}"

        message = record.getMessage()
        return f"{levelname}[{location}{funcname}]{message}"


def make_logger(output_dir: Path):
    log_file = output_dir / "result.log"

    formatter = VisibleColorFormatter()

    handlers = [
        logging.FileHandler(log_file),  # plain log file
        logging.StreamHandler(),  # color to terminal
    ]

    # Only apply color formatter to terminal
    handlers[1].setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=handlers)

    logger = logging.getLogger(__name__)
    return logger
