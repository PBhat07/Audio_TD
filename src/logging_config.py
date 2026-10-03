import logging
import os


def configure_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
) -> None:
    """Configure console and file logging for the application."""

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "audio_td.log")

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Prevent duplicate handlers if configure_logging()
    # is called more than once.
    root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)