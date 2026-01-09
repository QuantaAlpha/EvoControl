import logging
from pathlib import Path


def get_se_logger(
    name: str,
    file_path: str | Path,
    emoji: str | None = None,
    level: int = logging.INFO,
    also_stream: bool = True,
) -> logging.Logger:
    """Create or reconfigure a logger that writes to a specific file and optionally to the terminal.

    - Ensures parent directory exists
    - If the logger already exists, rebind its FileHandler to the requested file when different
    - Keeps a single StreamHandler (if requested) to avoid duplicates
    - Disables propagate to prevent duplicate logs to root handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    target_path = str(file_path)

    # Ensure directory exists
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If path handling fails, we still proceed; FileHandler may raise later
        pass

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if emoji:
        fmt = f"{emoji} " + fmt
    formatter = logging.Formatter(fmt)

    # Check if this logger already has handlers - if so, return it as-is
    # This prevents adding duplicate handlers when the same logger is initialized multiple times
    if logger.handlers:
        return logger

    # Check existing handlers for this specific file path across ALL loggers
    # This prevents multiple loggers from writing to the same file
    existing_file_handlers = []
    for logger_name in logging.root.manager.loggerDict:
        existing_logger = logging.getLogger(logger_name)
        for handler in existing_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                try:
                    if getattr(handler, "baseFilename", None) == target_path:
                        existing_file_handlers.append(handler)
                except Exception:
                    continue

    # Attach file handler
    fh = logging.FileHandler(target_path, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(level)
    logger.addHandler(fh)

    # Attach stream handler if requested
    if also_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(level)
        logger.addHandler(sh)

    # Prevent duplicate logging via root
    logger.propagate = False
    return logger
