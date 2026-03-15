from __future__ import annotations

import logging
import sys
from typing import Optional

try:  # pragma: no cover - import guard
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that plays nicely with tqdm progress bars.

    In notebooks and terminals this avoids garbling progress bars when logs are emitted
    during a tqdm loop.
    """

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        try:
            msg = self.format(record)
            if tqdm is not None:
                tqdm.write(msg)
            else:
                self.stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


def make_logger(
    name: str,
    *,
    level: int = logging.INFO,
    use_tqdm_handler: bool = True,
    reset_handlers: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if reset_handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    if not logger.handlers:
        handler: logging.Handler
        if use_tqdm_handler:
            handler = TqdmLoggingHandler(stream=sys.stdout)
        else:
            handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    return logger


def get_logger(name: str = "triage", level: int = logging.INFO) -> logging.Logger:
    return make_logger(name, level=level, use_tqdm_handler=True, reset_handlers=False)


class NullLogger:
    def debug(self, *args, **kwargs):
        return None

    info = warning = error = exception = debug


def coalesce_logger(logger: Optional[logging.Logger], name: str) -> logging.Logger:
    if logger is not None:
        return logger
    return get_logger(name)
