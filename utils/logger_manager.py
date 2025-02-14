import logging

class LoggerManager:
    def __init__(self, file_name):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        self.console_handler.setFormatter(console_format)
        if not self.logger.hasHandlers():
            self.logger.addHandler(self.console_handler)

        self.file_handler = logging.FileHandler(file_name, mode="w", encoding="utf-8")
        self.file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(message)s')
        self.file_handler.setFormatter(file_format)

    def write(self, string):
        self.logger.info(string)
