import logging

# Configura el logger
logger = logging.getLogger("debugger")
logger.setLevel(logging.DEBUG)  # Puedes cambiar a WARNING para desactivarlo

# Handler rápido a consola (solo una vez)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)