import pygame

def alarm(sound_file):
	sound_file = sound_file

	pygame.mixer.init()
	pygame.mixer.music.load(sound_file)
	pygame.mixer.music.play()

	clock = pygame.time.Clock()
	while pygame.mixer.music.get_busy():
		clock.tick(100)
	pygame.mixer.quit()
