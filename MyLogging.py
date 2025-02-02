from logging import getLogger, Formatter, handlers, DEBUG, ERROR, INFO

class FileLogger:
	def __init__(self, info_file='./log/app.log', debug_file='./log/debug.log', error_file='./log/error.log'):
		self.logger = getLogger(__name__)
		self.logger.setLevel(DEBUG)
		formatter = Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
		# DEBUGレベルハンドラ定義
		info_handler = handlers.RotatingFileHandler(
			filename=info_file,
			mode='w',
			backupCount=3,
		)
		info_handler.setFormatter(formatter)
		info_handler.setLevel(INFO)
		self.logger.addHandler(info_handler)
		# DEBUGレベルハンドラ定義
		debug_handler = handlers.RotatingFileHandler(
			filename=debug_file,
			mode='w',
			backupCount=3,
		)
		debug_handler.setFormatter(formatter)
		debug_handler.setLevel(DEBUG)
		self.logger.addHandler(debug_handler)
		# ERRORレベルハンドラ
		error_handler = handlers.RotatingFileHandler(
			filename=error_file,
			mode='w',
			backupCount=3
		)
		error_handler.setFormatter(formatter)
		error_handler.setLevel(ERROR)
		self.logger.addHandler(error_handler)

	def debug(self, msg):
		self.logger.debug(msg)

	def info(self, msg):
		self.logger.info(msg)

	def warn(self, msg):
		self.logger.warning(msg)

	def error(self, msg):
		self.logger.error(msg)

	def critical(self, msg):
		self.logger.critical(msg)
