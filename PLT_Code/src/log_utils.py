# Imports
import logging
import os
from datetime import datetime

# Classes
class Logger:
    def __init__(self, logger_name, log_file='execution', log_dir='../logs/', log_level='debug', show_log=True):
        self.logger = logging.getLogger(logger_name)
        self.config(log_file=log_file, log_dir=log_dir, log_level=log_level, show_log=show_log)
        pass

    def config(self, log_file='execution', log_dir = '../logs/', log_level='debug', show_log=True):
        now = datetime.now()
        log_fname = f'{log_dir}{log_file}_log_{str(now).replace(" ", "_")}.log'
        self.log_fname = log_fname.split('/')[-1]

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_level = logging.getLevelName(log_level.upper())
        self.logger.setLevel(log_level)

        log_formatter = logging.Formatter('%(levelname)s:%(name)s:File[%(filename)s]:'
                                          'Function[%(funcName)s]:Line[%(lineno)s]:: %(message)s')

        log_file_handler = logging.FileHandler(log_fname)
        log_file_handler.setFormatter(log_formatter)

        log_stream_handler = logging.StreamHandler()
        log_stream_handler.setFormatter(log_formatter)

        self.logger.addHandler(log_file_handler)
        if show_log:
            self.logger.addHandler(log_stream_handler)

        return self.logger

    def get_logger(self):
        return self.logger


# Main
if __name__ == '__main__':
    pass