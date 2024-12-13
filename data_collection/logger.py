import logging

def configure_log():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s -%(message)s",
                        handlers=[logging.StreamHandler()])
    return logging.getLogger("deep_learning_project")