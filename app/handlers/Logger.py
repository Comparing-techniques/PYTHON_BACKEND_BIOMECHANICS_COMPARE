import logging

# Configuración del logger
logger = logging.getLogger("debugger")
logger.setLevel(logging.DEBUG)

# Handler rápido a consola
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)