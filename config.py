import pygame # Added for toggle_music

# Configuration file for the game

# Sound settings
MUSIC_ENABLED = True
SFX_ENABLED = True

# Screen dimensions
WIDTH = 800
HEIGHT = 600

# Game settings
FPS = 60
NUM_ROOMS = 10
BULLET_INTERVAL = 200  # milliseconds
RHYTHM_BEAT_INTERVAL = 500  # milliseconds
ROTATION_SPEED = 5
THRUST = 0.1
MAX_SPEED = 5
COLLECTION_TIME = 5000  # milliseconds

# Sound effects (now using integer IDs)
SFX_SHOOT = 10
SFX_TELEPORT = 1
SFX_MINE_DROP = 2
SFX_AUTO_FIRE = 3
SFX_HIT_PLAYER = 9
SFX_HIT_ENEMY = 13
SFX_COUNTDOWN = 14
SFX_BUY_SUCCESS = 6
SFX_BUY_FAIL = 7
SFX_MOVE_LEFT = 22
SFX_MOVE_RIGHT = 20
SFX_MOVE_UP = 19
SFX_MOVE_DOWN = 18
SFX_STARMAP_OPEN = 23
SFX_STARMAP_TRAVEL = 24
SFX_STARMAP_CLOSE = 25

# Game states (changed back to strings)
STATE_PRE_DIALOG = 'PRE_DIALOG'
STATE_COMBAT = 'COMBAT'
STATE_COLLECTION = 'COLLECTION'
STATE_POST_DIALOG = 'POST_DIALOG'
STATE_STORE = 'STORE'
STATE_STAR_MAP = 'STAR_MAP'
STATE_END = 'END'

# Functions to toggle music and SFX
def toggle_music():
    global MUSIC_ENABLED
    MUSIC_ENABLED = not MUSIC_ENABLED
    if MUSIC_ENABLED:
        pygame.mixer.music.unpause()
    else:
        pygame.mixer.music.pause()

def toggle_sfx():
    global SFX_ENABLED
    SFX_ENABLED = not SFX_ENABLED
