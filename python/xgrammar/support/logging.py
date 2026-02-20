"""
Logging support for XGrammar. It derives from Python's logging module, and in the future,
it can be easily replaced by other logging modules such as structlog.
"""

import logging


def enable_logging():
    """Enable XGrammar's default logging format for the xgrammar logger only."""

    log = logging.getLogger("xgrammar")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "[{asctime}] {levelname} {filename}:{lineno}: {message}",
                datefmt="%Y-%m-%d %H:%M:%S",
                style="{",
            )
        )
        log.addHandler(handler)


def getLogger(name: str):  # pylint: disable=invalid-name
    """Get a logger according to the given name"""
    return logging.getLogger(name)
