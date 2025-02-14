class LogToFile:
    """Context manager to write temporarily only on file."""
    def __init__(self, logger_manager):
        self.logger_manager = logger_manager
        self.logger = logger_manager.logger
        self.console_handler = logger_manager.console_handler
        self.file_handler = logger_manager.file_handler
    
    def __enter__(self):
        if self.console_handler in self.logger.handlers:
            self.logger.removeHandler(self.console_handler)
        if self.file_handler not in self.logger.handlers:
            self.logger.addHandler(self.file_handler)
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.file_handler in self.logger.handlers:
            self.logger.removeHandler(self.file_handler)
        if self.console_handler not in self.logger.handlers:
            self.logger.addHandler(self.console_handler)