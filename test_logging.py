import logging

# 创建一个日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 创建一个控制台处理器，并设置日志级别为DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 创建一个文件处理器，并设置日志级别为INFO
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

# 定义日志输出格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 示例日志信息
logger.debug('Debug message')
logger.info('Info message')
logger.warning('Warning message')
logger.error('Error message')
logger.critical('Critical message')
