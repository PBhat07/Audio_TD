import logging
import os


def configure_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
) -> None:
    """Configure console, application, and GPU memory logging."""

    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "audio_td.log")
    gpu_log_file = os.path.join(log_dir, "gpu_memory.log")

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # ---------------------------------------------------------
    # Console handler
    # ---------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # ---------------------------------------------------------
    # Main application log
    # ---------------------------------------------------------
    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # ---------------------------------------------------------
    # Root logger
    # ---------------------------------------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Prevent duplicate handlers if configure_logging()
    # is called more than once.
    root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # ---------------------------------------------------------
    # Dedicated GPU memory logger
    # ---------------------------------------------------------
    gpu_logger = logging.getLogger("gpu_memory")

    gpu_logger.setLevel(log_level)
    gpu_logger.handlers.clear()
    gpu_logger.propagate = False

    gpu_console_handler = logging.StreamHandler()
    gpu_console_handler.setFormatter(formatter)

    gpu_file_handler = logging.FileHandler(
        os.path.join(log_dir, "gpu_memory.log"),
        encoding="utf-8",
    )
    gpu_file_handler.setFormatter(formatter)

    gpu_logger.addHandler(gpu_console_handler)
    gpu_logger.addHandler(gpu_file_handler)