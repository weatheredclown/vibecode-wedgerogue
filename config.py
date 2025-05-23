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

# Sound effects (placeholders, replace with actual sound file paths or integer IDs if your system uses them)
SFX_SHOOT = "shoot.wav" # Or an integer ID like 10
SFX_TELEPORT = "teleport.wav" # Or an integer ID like 1
SFX_MINE_DROP = "mine_drop.wav" # Or an integer ID like 2
SFX_AUTO_FIRE = "auto_fire.wav" # Or an integer ID like 3
SFX_HIT_PLAYER = "hit_player.wav" # Or an integer ID like 9
SFX_HIT_ENEMY = "hit_enemy.wav" # Or an integer ID like 13
SFX_COUNTDOWN = "countdown.wav" # Or an integer ID like 14
SFX_BUY_SUCCESS = "buy_success.wav" # Or an integer ID like 6
SFX_BUY_FAIL = "buy_fail.wav" # Or an integer ID like 7
SFX_MOVE_LEFT = "move_left.wav" # Or an integer ID like 22
SFX_MOVE_RIGHT = "move_right.wav" # Or an integer ID like 20
SFX_MOVE_UP = "move_up.wav" # Or an integer ID like 19
SFX_MOVE_DOWN = "move_down.wav" # Or an integer ID like 18
SFX_STARMAP_OPEN = "starmap_open.wav" # Or an integer ID like 23
SFX_STARMAP_TRAVEL = "starmap_travel.wav" # Or an integer ID like 24
SFX_STARMAP_CLOSE = "starmap_close.wav" # Or an integer ID like 25

# Game states
STATE_PRE_DIALOG = 0
STATE_COMBAT = 1
STATE_COLLECTION = 2
STATE_POST_DIALOG = 3
STATE_STORE = 4
STATE_STAR_MAP = 5
STATE_END = 6

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
