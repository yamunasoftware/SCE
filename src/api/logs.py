### API LOGGING ###

from logging.handlers import TimedRotatingFileHandler
import logging

# Logging Setup:
def setup_logging():
  handler = TimedRotatingFileHandler(
    filename='/main/logs/SCE.log',
    when='midnight',
    interval=1,
    backupCount=30,
    encoding='utf-8'
  )
  handler.suffix = '%Y-%m-%d'
  
  logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s %(message)s',
    handlers=[handler]
  )

# Log Info Level:
def log_info(message):
  logging.info(message)

# Log Error Level:
def log_error(message):
  logging.error(message)