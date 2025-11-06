
import os
import logging


class AsyncHandler(logging.Handler):
    """
    自定义异步日志处理器。
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            self.flush(msg)
        except Exception:
            self.handleError(record)

    def flush(self, msg):
        raise NotImplementedError("Subclasses should implement this method.")

class AsyncFileHandler(AsyncHandler):
    """
    异步文件日志处理器。
    """
    def __init__(self, filename, mode='w+', encoding=None, delay=False):
        super().__init__()
        self.filename = filename
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        self.file = None

    def flush(self, msg):
        if self.file is None:
            self.file = open(self.filename, self.mode, encoding=self.encoding, buffering=1)
        self.file.write(msg + '\n')

    def close(self):
        super().close()
        if self.file is not None:
            self.file.close()
            self.file = None

class LoggerBase:

    # 定义日志格式
    LOG_FORMAT = '%(asctime)s [%(round)s][%(identity)s] [%(action)s] %(message)s'
    DATE_FORMAT = '%Y/%m/%d-%H/%M/%S'

    def __init__(self, log_name, log_file_path):
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # 创建异步文件处理器
        async_file_handler = AsyncFileHandler(os.path.join(log_file_path, log_name))
        async_file_handler.setLevel(logging.DEBUG)

        # 创建格式化器
        formatter = logging.Formatter(LoggerBase.LOG_FORMAT, datefmt=LoggerBase.DATE_FORMAT)

        # 将格式化器应用到处理器上
        async_file_handler.setFormatter(formatter)

        # 将处理器添加到logger
        if not self.logger.handlers:  # 防止多次添加处理器
            self.logger.addHandler(async_file_handler)

    def log(self, round, identity, action, message):
        round = f"Round-{round}".zfill(4)
        extra = {'round': round,'identity': identity, 'action': action}
        self.logger.log(logging.INFO, message, extra=extra)

