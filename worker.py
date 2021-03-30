import threading
from alarm import alarm

class Worker(threading.Thread):
	def __init__(self):
		super().__init__()
	
	def run(self):
		alarm('sample.mp3')


