import asyncio
import os
import math
import random
import sys
#import numpy as np
import pygame
from pygame.math import Vector2

##############################################################################
# GLOBAL SETTINGS for music/SFX toggles (for Options Menu)
##############################################################################
MUSIC_ENABLED = False   # if False => no background music
SFX_ENABLED   = False   # if False => sfx_system won’t actually .play()

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

##############################################################################
# (Optional) Echo & Distortion
##############################################################################
def apply_echo(samples, sample_rate: int, delay_seconds=0.2, feedback=0.5):
    delay_samples = int(delay_seconds * sample_rate)
    if delay_samples <= 0:
        return samples
    processed = samples.astype(np.int32)
    n = len(samples)
    for i in range(delay_samples, n):
        processed[i] += int(processed[i - delay_samples] * feedback)
    processed = np.clip(processed, -32768, 32767).astype(np.int16)
    return processed

def apply_distortion(samples, threshold=0.3):
    max_amp = 32767
    clip_value = int(threshold * max_amp)
    s32 = samples.astype(np.int32)
    s32[s32 >  clip_value] =  clip_value
    s32[s32 < -clip_value] = -clip_value
    return s32.astype(np.int16)

###########################################################
# 2) generate_fm_sound: now using envelope times + pitch
###########################################################
def generate_fm_sound(cf:float,mf:float,mi:float,du:float,
                      sr:int=44100, echo=False,ed=0.2,ef=0.5,
                      dist=False,th=0.3,
                      at=0.08,d1=0.10,d2=0.10,rl=0.50,
                      pt="dn"):
    """
    FM with amplitude envelope (attack,decay1,decay2,release) 
    + 'pt' controlling pitch trajectory:
       'dn' => descending
       'up' => ascending
       'sw' => swoop/sinus
       'st' => step-based
    """
    from math import pi,sin

    num_samps = int(sr*du)
    t = np.linspace(0,du,num_samps,endpoint=False)

    # 1) pitch factor
    if pt=="dn":
        pitch_factor=np.linspace(1.0,0.2,num_samps)
    elif pt=="up":
        pitch_factor=np.linspace(0.3,1.2,num_samps)
    elif pt=="sw":
        pitch_factor=1.0+0.4*np.sin(2*pi*1.0*t)
    elif pt=="st":
        step_sz=int(0.3*sr)
        pitch_factor=np.ones(num_samps)
        for i in range(num_samps):
            st_id=i//step_sz
            pitch_factor[i]=1.0+0.1*st_id
    else:
        pitch_factor=np.linspace(1.0,0.2,num_samps)

    # 2) freq-based FM
    cphase=2*pi*(cf*pitch_factor)*t
    mphase=2*pi*(mf*pitch_factor)*t
    fm=np.sin(cphase+mi*np.sin(mphase))

    # 3) amplitude envelope
    a_samps=int(at*sr)
    d1_samps=int(d1*sr)
    d2_samps=int(d2*sr)
    r_samps=int(rl*sr)
    total_env=a_samps+d1_samps+d2_samps+r_samps
    if total_env>num_samps:
        sc=num_samps/total_env
        a_samps=int(a_samps*sc); d1_samps=int(d1_samps*sc)
        d2_samps=int(d2_samps*sc); r_samps=int(r_samps*sc)
    env = np.concatenate([
      np.linspace(0,1,a_samps,endpoint=False),
      np.linspace(1,0.6,d1_samps,endpoint=False),
      np.linspace(0.6,0.3,d2_samps,endpoint=False),
      np.linspace(0.3,0,r_samps,endpoint=False)
    ])
    if len(env)<num_samps:
        env=np.pad(env,(0,num_samps-len(env)),'constant')
    elif len(env)>num_samps:
        env=env[:num_samps]

    samples=(fm*env*32767).astype(np.int16)
    # 4) Distortion/echo
    if dist: samples=apply_distortion(samples,th)
    if echo: samples=apply_echo(samples,sr,ed,ef)
    return samples

###########################################################
# 3) A small helper that reads from short param dictionary
###########################################################
def generate_fm_sound_from_dict(cfg):
    """
    Example keys:
    {
      'cf':80, 'mf':40, 'mi':8, 'du':1.8, 
      'ec':False, 'ed':0.2, 'ef':0.5,
      'dist':False, 'th':0.3,
      'at':0.08, 'd1':0.10, 'd2':0.10, 'rl':0.50,
      'pt':'dn'
    }
    """
    return generate_fm_sound(
      cf=cfg.get('cf',220),mf=cfg.get('mf',110),mi=cfg.get('mi',2),du=cfg.get('du',1.0),
      sr=44100,
      echo=cfg.get('ec',False),ed=cfg.get('ed',0.2),ef=cfg.get('ef',0.5),
      dist=cfg.get('dist',False),th=cfg.get('th',0.3),
      at=cfg.get('at',0.08),d1=cfg.get('d1',0.10),d2=cfg.get('d2',0.10),rl=cfg.get('rl',0.5),
      pt=cfg.get('pt','dn')
    )

##############################################################################
# 25 SFX Presets with longer durations & nuanced envelopes
##############################################################################
DESC_FAMILY = [
  # We'll vary carrier freq, mod freq, echo, distortion, envelope a bit
  # We'll do 16 of these:
  {'cf':80,'mf':40,'mi':8,'du':1.8,'pt':'dn','at':0.05,'d1':0.1,'d2':0.1,'rl':0.4,'ec':False,'dist':False},
  {'cf':85,'mf':20,'mi':12,'du':2.0,'pt':'dn','at':0.08,'d1':0.1,'d2':0.2,'rl':0.5,'ec':True,'ed':0.2,'ef':0.5,'dist':False},
  {'cf':90,'mf':40,'mi':2,'du':1.7,'pt':'dn','at':0.1,'d1':0.15,'d2':0.12,'rl':0.5,'ec':False,'dist':True,'th':0.3},
  {'cf':100,'mf':60,'mi':6,'du':2.1,'pt':'dn','at':0.08,'d1':0.1,'d2':0.15,'rl':0.4,'ec':False,'dist':False},
  {'cf':140,'mf':10,'mi':10,'du':2.3,'pt':'dn','at':0.12,'d1':0.12,'d2':0.15,'rl':0.5,'ec':True,'ed':0.25,'ef':0.4,'dist':False},
  {'cf':100,'mf':12,'mi':3,'du':1.9,'pt':'dn','at':0.05,'d1':0.08,'d2':0.08,'rl':0.6,'ec':False,'dist':True,'th':0.4},
  {'cf':70,'mf':25,'mi':8,'du':1.5,'pt':'dn','at':0.06,'d1':0.1,'d2':0.1,'rl':0.4,'ec':True,'ed':0.2,'ef':0.6,'dist':False},
  {'cf':160,'mf':30,'mi':14,'du':2.0,'pt':'dn','at':0.08,'d1':0.1,'d2':0.1,'rl':0.6,'ec':False,'dist':False},
  {'cf':75,'mf':10,'mi':20,'du':1.8,'pt':'dn','at':0.05,'d1':0.12,'d2':0.12,'rl':0.4,'ec':False,'dist':False},
  {'cf':125,'mf':10,'mi':9,'du':2.4,'pt':'dn','at':0.1,'d1':0.15,'d2':0.15,'rl':0.5,'ec':True,'ed':0.28,'ef':0.4,'dist':True,'th':0.3},
  {'cf':180,'mf':70,'mi':5,'du':1.7,'pt':'dn','at':0.12,'d1':0.15,'d2':0.15,'rl':0.4,'ec':False,'dist':True,'th':0.4},
  {'cf':200,'mf':80,'mi':2,'du':2.2,'pt':'dn','at':0.08,'d1':0.08,'d2':0.2,'rl':0.5,'ec':False,'dist':False},
  {'cf':50,'mf':10,'mi':15,'du':1.9,'pt':'dn','at':0.08,'d1':0.1,'d2':0.2,'rl':0.4,'ec':True,'ed':0.15,'ef':0.5,'dist':False},
  {'cf':140,'mf':70,'mi':16,'du':2.0,'pt':'dn','at':0.1,'d1':0.1,'d2':0.2,'rl':0.5,'ec':False,'dist':True,'th':0.3},
  {'cf':210,'mf':30,'mi':8,'du':1.6,'pt':'dn','at':0.05,'d1':0.08,'d2':0.1,'rl':0.6,'ec':False,'dist':False},
  {'cf':95,'mf':90,'mi':10,'du':2.3,'pt':'dn','at':0.07,'d1':0.1,'d2':0.1,'rl':0.5,'ec':True,'ed':0.18,'ef':0.4,'dist':True,'th':0.4},
]

ASC_FAMILY = [
  {'cf':80,'mf':10,'mi':8,'du':1.8,'pt':'up','at':0.05,'d1':0.1,'d2':0.1,'rl':0.4,'ec':False,'dist':False},
  {'cf':85,'mf':20,'mi':12,'du':2.0,'pt':'up','at':0.08,'d1':0.1,'d2':0.2,'rl':0.5,'ec':True,'ed':0.2,'ef':0.5,'dist':False},
  {'cf':90,'mf':40,'mi':2,'du':1.7,'pt':'up','at':0.1,'d1':0.15,'d2':0.12,'rl':0.5,'ec':False,'dist':True,'th':0.3},
  {'cf':100,'mf':60,'mi':6,'du':2.1,'pt':'up','at':0.08,'d1':0.1,'d2':0.15,'rl':0.4,'ec':False,'dist':False},
  {'cf':140,'mf':10,'mi':10,'du':2.3,'pt':'up','at':0.12,'d1':0.12,'d2':0.15,'rl':0.5,'ec':True,'ed':0.25,'ef':0.4,'dist':False},
  {'cf':100,'mf':12,'mi':3,'du':1.9,'pt':'up','at':0.05,'d1':0.08,'d2':0.08,'rl':0.6,'ec':False,'dist':True,'th':0.4},
  {'cf':70,'mf':25,'mi':8,'du':1.5,'pt':'up','at':0.06,'d1':0.1,'d2':0.1,'rl':0.4,'ec':True,'ed':0.2,'ef':0.6,'dist':False},
  {'cf':160,'mf':30,'mi':14,'du':2.0,'pt':'up','at':0.08,'d1':0.1,'d2':0.1,'rl':0.6,'ec':False,'dist':False},
  {'cf':75,'mf':10,'mi':20,'du':1.8,'pt':'up','at':0.05,'d1':0.12,'d2':0.12,'rl':0.4,'ec':False,'dist':False},
  {'cf':125,'mf':10,'mi':9,'du':2.4,'pt':'up','at':0.1,'d1':0.15,'d2':0.15,'rl':0.5,'ec':True,'ed':0.28,'ef':0.4,'dist':True,'th':0.3},
  {'cf':180,'mf':70,'mi':5,'du':1.7,'pt':'up','at':0.12,'d1':0.15,'d2':0.15,'rl':0.4,'ec':False,'dist':True,'th':0.4},
  {'cf':200,'mf':80,'mi':2,'du':2.2,'pt':'up','at':0.08,'d1':0.08,'d2':0.2,'rl':0.5,'ec':False,'dist':False},
  {'cf':50,'mf':10,'mi':15,'du':1.9,'pt':'up','at':0.08,'d1':0.1,'d2':0.2,'rl':0.4,'ec':True,'ed':0.15,'ef':0.5,'dist':False},
  {'cf':140,'mf':70,'mi':16,'du':2.0,'pt':'up','at':0.1,'d1':0.1,'d2':0.2,'rl':0.5,'ec':False,'dist':True,'th':0.3},
  {'cf':210,'mf':30,'mi':8,'du':1.6,'pt':'up','at':0.05,'d1':0.08,'d2':0.1,'rl':0.6,'ec':False,'dist':False},
  {'cf':95,'mf':90,'mi':10,'du':2.3,'pt':'up','at':0.07,'d1':0.1,'d2':0.1,'rl':0.5,'ec':True,'ed':0.18,'ef':0.4,'dist':True,'th':0.4},
]

SWOOP_FAMILY = [
  {'cf':80,'mf':40,'mi':8,'du':1.8,'pt':'sw','at':0.05,'d1':0.1,'d2':0.1,'rl':0.4,'ec':False,'dist':False},
  {'cf':85,'mf':20,'mi':12,'du':2.0,'pt':'sw','at':0.08,'d1':0.1,'d2':0.2,'rl':0.5,'ec':True,'ed':0.2,'ef':0.5,'dist':False},
  {'cf':90,'mf':40,'mi':2,'du':1.7,'pt':'sw','at':0.1,'d1':0.15,'d2':0.12,'rl':0.5,'ec':False,'dist':True,'th':0.3},
  {'cf':100,'mf':60,'mi':6,'du':2.1,'pt':'sw','at':0.08,'d1':0.1,'d2':0.15,'rl':0.4,'ec':False,'dist':False},
  {'cf':140,'mf':10,'mi':10,'du':2.3,'pt':'sw','at':0.12,'d1':0.12,'d2':0.15,'rl':0.5,'ec':True,'ed':0.25,'ef':0.4,'dist':False},
  {'cf':100,'mf':12,'mi':3,'du':1.9,'pt':'sw','at':0.05,'d1':0.08,'d2':0.08,'rl':0.6,'ec':False,'dist':True,'th':0.4},
  {'cf':70,'mf':25,'mi':8,'du':1.5,'pt':'sw','at':0.06,'d1':0.1,'d2':0.1,'rl':0.4,'ec':True,'ed':0.2,'ef':0.6,'dist':False},
  {'cf':160,'mf':30,'mi':14,'du':2.0,'pt':'sw','at':0.08,'d1':0.1,'d2':0.1,'rl':0.6,'ec':False,'dist':False},
  {'cf':75,'mf':10,'mi':20,'du':1.8,'pt':'sw','at':0.05,'d1':0.12,'d2':0.12,'rl':0.4,'ec':False,'dist':False},
  {'cf':125,'mf':10,'mi':9,'du':2.4,'pt':'sw','at':0.1,'d1':0.15,'d2':0.15,'rl':0.5,'ec':True,'ed':0.28,'ef':0.4,'dist':True,'th':0.3},
  {'cf':180,'mf':70,'mi':5,'du':1.7,'pt':'sw','at':0.12,'d1':0.15,'d2':0.15,'rl':0.4,'ec':False,'dist':True,'th':0.4},
  {'cf':200,'mf':80,'mi':2,'du':2.2,'pt':'sw','at':0.08,'d1':0.08,'d2':0.2,'rl':0.5,'ec':False,'dist':False},
  {'cf':50,'mf':10,'mi':15,'du':1.9,'pt':'sw','at':0.08,'d1':0.1,'d2':0.2,'rl':0.4,'ec':True,'ed':0.15,'ef':0.5,'dist':False},
  {'cf':140,'mf':70,'mi':16,'du':2.0,'pt':'sw','at':0.1,'d1':0.1,'d2':0.2,'rl':0.5,'ec':False,'dist':True,'th':0.3},
  {'cf':210,'mf':30,'mi':8,'du':1.6,'pt':'sw','at':0.05,'d1':0.08,'d2':0.1,'rl':0.6,'ec':False,'dist':False},
  {'cf':95,'mf':90,'mi':10,'du':2.3,'pt':'sw','at':0.07,'d1':0.1,'d2':0.1,'rl':0.5,'ec':True,'ed':0.18,'ef':0.4,'dist':True,'th':0.4},
]

STEP_FAMILY = [
  {'cf':80,'mf':40,'mi':8,'du':1.8,'pt':'st','at':0.05,'d1':0.1,'d2':0.1,'rl':0.4,'ec':False,'dist':False},
  {'cf':85,'mf':20,'mi':12,'du':2.0,'pt':'st','at':0.08,'d1':0.1,'d2':0.2,'rl':0.5,'ec':True,'ed':0.2,'ef':0.5,'dist':False},
  {'cf':90,'mf':40,'mi':2,'du':1.7,'pt':'st','at':0.1,'d1':0.15,'d2':0.12,'rl':0.5,'ec':False,'dist':True,'th':0.3},
  {'cf':100,'mf':60,'mi':6,'du':2.1,'pt':'st','at':0.08,'d1':0.1,'d2':0.15,'rl':0.4,'ec':False,'dist':False},
  {'cf':140,'mf':10,'mi':10,'du':2.3,'pt':'st','at':0.12,'d1':0.12,'d2':0.15,'rl':0.5,'ec':True,'ed':0.25,'ef':0.4,'dist':False},
  {'cf':100,'mf':12,'mi':3,'du':1.9,'pt':'st','at':0.05,'d1':0.08,'d2':0.08,'rl':0.6,'ec':False,'dist':True,'th':0.4},
  {'cf':70,'mf':25,'mi':8,'du':1.5,'pt':'st','at':0.06,'d1':0.1,'d2':0.1,'rl':0.4,'ec':True,'ed':0.2,'ef':0.6,'dist':False},
  {'cf':160,'mf':30,'mi':14,'du':2.0,'pt':'st','at':0.08,'d1':0.1,'d2':0.1,'rl':0.6,'ec':False,'dist':False},
  {'cf':75,'mf':10,'mi':20,'du':1.8,'pt':'st','at':0.05,'d1':0.12,'d2':0.12,'rl':0.4,'ec':False,'dist':False},
  {'cf':125,'mf':10,'mi':9,'du':2.4,'pt':'st','at':0.1,'d1':0.15,'d2':0.15,'rl':0.5,'ec':True,'ed':0.28,'ef':0.4,'dist':True,'th':0.3},
  {'cf':180,'mf':70,'mi':5,'du':1.7,'pt':'st','at':0.12,'d1':0.15,'d2':0.15,'rl':0.4,'ec':False,'dist':True,'th':0.4},
  {'cf':200,'mf':80,'mi':2,'du':2.2,'pt':'st','at':0.08,'d1':0.08,'d2':0.2,'rl':0.5,'ec':False,'dist':False},
  {'cf':50,'mf':10,'mi':15,'du':1.9,'pt':'st','at':0.08,'d1':0.1,'d2':0.2,'rl':0.4,'ec':True,'ed':0.15,'ef':0.5,'dist':False},
  {'cf':140,'mf':70,'mi':16,'du':2.0,'pt':'st','at':0.1,'d1':0.1,'d2':0.2,'rl':0.5,'ec':False,'dist':True,'th':0.3},
  {'cf':210,'mf':30,'mi':8,'du':1.6,'pt':'st','at':0.05,'d1':0.08,'d2':0.1,'rl':0.6,'ec':False,'dist':False},
  {'cf':95,'mf':90,'mi':10,'du':2.3,'pt':'st','at':0.07,'d1':0.1,'d2':0.1,'rl':0.5,'ec':True,'ed':0.18,'ef':0.4,'dist':True,'th':0.4},
]

# Flatten them all into a single 64-item list
SFX_PRESETS = DESC_FAMILY + ASC_FAMILY + SWOOP_FAMILY + STEP_FAMILY

##############################################################################
# SfxSystem class
##############################################################################
class SfxSystem:
    def __init__(self):
        # Initialize Pygame mixer for 44.1k, 16-bit mono
        if SFX_ENABLED or MUSIC_ENABLED:
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        self.sounds = []
        # Pre-generate each preset for quick playback
        for i, params in enumerate(SFX_PRESETS):
            if SFX_ENABLED:
                raw_samples = generate_fm_sound_from_dict(params)
                snd = pygame.mixer.Sound(buffer=raw_samples.tobytes())
                self.sounds.append(snd)

    def play_sound(self, index):
        # only play if SFX_ENABLED
        if not SFX_ENABLED:
            return
        if 0 <= index < len(self.sounds):
            self.sounds[index].play()
        else:
            print(f"Invalid sfx index {index}")

sfx_system = SfxSystem()

################################################################################
# GAME CONSTANTS & DATA
################################################################################
WIDTH, HEIGHT = 1024, 768
FPS = 60
NUM_ROOMS = 5
BULLET_INTERVAL = 60
RHYTHM_BEAT_INTERVAL = 120
ROTATION_SPEED = 3
THRUST = 0.2
MAX_SPEED = 8
COLLECTION_TIME = 10  # seconds to gather loot

# Some basic categories
SFX_SHOOT       = 10
SFX_TELEPORT    = 1
SFX_MINE_DROP   = 2
SFX_AUTO_FIRE   = 3
SFX_HIT_PLAYER  = 9
SFX_HIT_ENEMY   = 13
SFX_COUNTDOWN   = 14
SFX_BUY_SUCCESS = 6
SFX_BUY_FAIL    = 7
SFX_MOVE_LEFT   = 22
SFX_MOVE_RIGHT  = 20
SFX_MOVE_UP     = 19
SFX_MOVE_DOWN   = 18

# Perhaps new ones for star map:
SFX_STARMAP_OPEN  = 23
SFX_STARMAP_TRAVEL= 24
SFX_STARMAP_CLOSE = 25

##############################################################################
# Screen Shake Globals
##############################################################################
SCREEN_SHAKE_MAG = 0
SCREEN_SHAKE_DECAY = 0.9  # how quickly the shake magnitude decays
def trigger_shake(amount=10):
    global SCREEN_SHAKE_MAG
    if amount>SCREEN_SHAKE_MAG:
        SCREEN_SHAKE_MAG = amount

##############################################################################
# ENEMY & BOSS DEFINITIONS
##############################################################################
ENEMY_TYPES = [
    ((255,   0,   0), 5,  5),
    ((255, 140,   0), 8,  10),
    ((255, 255,   0), 12, 15),
    ((  0, 255,   0), 15, 20),
    ((  0, 200, 200),20, 25)
]

class Enemy:
    def __init__(self, pos, is_boss=False):
        """
        :param pos: (x, y) spawn position
        :param is_boss: if True, this enemy is a 'boss' with higher health, bigger radius, etc.
        """
        self.pos = Vector2(pos)
        self.radius = 20
        self.vel = Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
        self.bullets = []
        self.angle = 0
        self.timer = 0

        # e.g. (color, max_health, points)
        etype = random.choice(ENEMY_TYPES)
        self.color = etype[0]       # Use same color for body fill AND loot
        self.max_health = etype[1]
        self.health = self.max_health
        self.points = etype[2]
        self.dropped_loot = False

        # If flagged as a boss, override some stats
        self.is_boss = is_boss
        if self.is_boss:
            # Example: drastically higher HP, bigger radius, special color, etc.
            self.max_health = 200
            self.health = self.max_health
            self.radius = 30
            self.color = (255, 80, 80)  # Boss color (you can change this)

        # Load sprite
        self.sprite = pygame.image.load("assets/enemy.png").convert_alpha()
        self.sprite_height = 40  # Match the height of the ship
        sprite_width = int(self.sprite.get_width() * (self.sprite_height / self.sprite.get_height()))
        self.sprite = pygame.transform.scale(self.sprite, (sprite_width, self.sprite_height))
        # self.sprite_alpha = 128  # optional transparency

    def update(self, global_frame, enemies):
        # normal movement
        self.pos += self.vel * 0.5
        self.pos = wrap_position(self.pos, WIDTH, HEIGHT)

        self.timer += 1

        # bullet pattern
        if self.timer % BULLET_INTERVAL == 0:
            # If boss, maybe shoot more bullets or do something unique
            bullet_step = 30 if not self.is_boss else 15  # boss fires more bullets?
            for angle_deg in range(0, 360, bullet_step):
                angle_rad = math.radians(angle_deg + self.timer * 5)
                bx = math.cos(angle_rad)
                by = math.sin(angle_rad)
                speed = 3 + 1 * math.sin(global_frame / 20)
                sfx_system.play_sound(10)
                self.bullets.append(
                    Bullet(self.pos, (bx * speed, by * speed), color=self.color)
                )

        # Update bullets
        for b in self.bullets:
            b.update(None)
        self.bullets = [b for b in self.bullets if not b.is_dead()]

    def draw(self, surface):
        # Diamond outline
        diamond_points = self.get_diamond_points()
        pygame.draw.polygon(surface, (255, 255, 255), diamond_points, width=2)

        # Fill portion for health
        fraction = max(0.0, min(1.0, self.health / self.max_health))
        if fraction > 0:
            fill_poly = self.get_health_fill_polygon(fraction)
            # fill with enemy color
            pygame.draw.polygon(surface, self.color, fill_poly)

        # draw bullets
        for b in self.bullets:
            b.draw(surface)

        # Draw the sprite
        rotated_sprite = pygame.transform.rotate(self.sprite, -self.angle)
        sprite_rect = rotated_sprite.get_rect(center=(self.pos.x, self.pos.y))
        # rotated_sprite.set_alpha(self.sprite_alpha)
        surface.blit(rotated_sprite, sprite_rect.topleft)

    def get_diamond_points(self):
        """
        Return 4 points in the shape of a diamond around self.pos, top->right->bottom->left.
        """
        x, y = self.pos
        r = self.radius
        return [
            (x,     y - r),  # top
            (x + r, y    ),  # right
            (x,     y + r),  # bottom
            (x - r, y    ),  # left
        ]

    def get_health_fill_polygon(self, fraction):
        """
        Returns the polygon that covers the bottom portion of the diamond up to
        'fraction' of the total diamond height (from bottom to top).
        fraction=1 => entire diamond, fraction=0 => none.
        """
        diamond = self.get_diamond_points()  # [top, right, bottom, left]
        top    = diamond[0]
        right  = diamond[1]
        bottom = diamond[2]
        left   = diamond[3]

        # We'll define a horizontal line at 'fill_y'
        fill_y = bottom[1] + fraction * (top[1] - bottom[1])

        edges = [
            (diamond[0], diamond[1]),
            (diamond[1], diamond[2]),
            (diamond[2], diamond[3]),
            (diamond[3], diamond[0]),
        ]

        fill_points = []
        for (p1, p2) in edges:
            if p1[1] >= fill_y:
                fill_points.append(p1)
            cross = self.edge_intersects_horizontal(p1, p2, fill_y)
            if cross is not None:
                fill_points.append(cross)

        # Sort by angle
        cx, cy = self.pos

        def angle_key(pt):
            dx = pt[0] - cx
            dy = pt[1] - cy
            return math.atan2(dy, dx)

        unique_pts = list(dict.fromkeys(fill_points))
        unique_pts.sort(key=angle_key)
        return unique_pts

    def edge_intersects_horizontal(self, p1, p2, line_y):
        """
        If segment p1->p2 crosses y=line_y, return intersection (x, line_y), else None.
        """
        x1, y1 = p1
        x2, y2 = p2
        if (y1 < line_y and y2 < line_y):
            return None
        if (y1 > line_y and y2 > line_y):
            return None
        if abs(y2 - y1) < 1e-9:
            return None

        t = (line_y - y1) / (y2 - y1)
        if 0 <= t <= 1:
            ix = x1 + t * (x2 - x1)
            return (ix, line_y)
        return None

    def hit_by(self, bullet):
        """
        Returns True if the bullet hits this enemy. Also checks if bullet is a 'mine'
        explosion so we can do a screen-shake. The code that *actually* kills the
        enemy or triggers drops is likely outside, but this can do side effects.
        """
        dist_sq = (self.pos - bullet.pos).length_squared()
        if dist_sq < (self.radius + bullet.radius)**2:
            # Check if it's an exploding mine => screen shake
            if bullet.exploding and bullet.bigger:
                # This bullet is presumably a 'mine' or large exploding bullet.
                # Trigger a global or external function to shake.
                if callable(globals().get("trigger_screen_shake")):
                    trigger_screen_shake(15)
                # or if you have a global variable, e.g. SCREEN_SHAKE_FRAMES
                # global SCREEN_SHAKE_FRAMES
                # SCREEN_SHAKE_FRAMES = 15

            return True
        return False

POWERUPS = [
    {"name": "Homing Bullets",         "cost": 50,  "desc": "Bullets steer slightly towards enemies."},
    {"name": "Better Homing Bullets",  "cost": 100, "desc": "Bullets steer more aggressively."},
    {"name": "Exploding Bullets",      "cost": 80,  "desc": "Bullets explode on impact."},
    {"name": "Bigger Bullets",         "cost": 60,  "desc": "Increase bullet size and maybe damage."},
    {"name": "Ship Shield",            "cost": 120, "desc": "Absorb some hits before taking damage."},
    {"name": "Ship Teleporter",        "cost": 150, "desc": "Press [T] to teleport short distance."},
    {"name": "Short-Range Autofire",   "cost": 90,  "desc": "Automatically fires at nearby enemies."},
    {"name": "Mines",                  "cost": 70,  "desc": "Press [M] to drop stationary mines."},
    {"name": "Homing Mines",           "cost": 130, "desc": "Mines [M] slowly chase enemies."},
]


def draw_text(surface, text, x, y, color=(255,255,255), font_size=24, center=False):
    font = pygame.font.SysFont(None, font_size)
    lines = text.split('\n')
    offset_y = 0
    for line in lines:
        rendered = font.render(line, True, color)
        if center:
            rect = rendered.get_rect(center=(x, y + offset_y))
        else:
            rect = rendered.get_rect(topleft=(x, y + offset_y))
        surface.blit(rendered, rect)
        offset_y += font_size + 2

def wrap_position(pos, width, height):
    x, y = pos
    if x < 0: 
        x = width
    elif x > width:
        x = 0
    if y < 0:
        y = height
    elif y > height:
        y = 0
    return Vector2(x, y)

################################################################################
# GAME ENTITIES
################################################################################
class FinalFlashingPressStart:
    """
    A final integrated script that:
    1) Shows a starfield
    2) Draws orbit rings at center
    3) Sparkles near the text
    4) "WEDGEROGUE" main text
    5) Flashes "PRESS SPACE"
    """

    # Colors
    COLOR_GOLD   = (255, 200,  50)
    COLOR_DARKER = (200,  30,  30)
    COLOR_BLACK  = (  0,   0,   0)
    COLOR_STARS  = (255, 200,  50)

    # Letter polygons dictionary
    LETTERS = {
        'C': [
            [
                (15,0),
                (3,0),
                (0,3),
                (0,37),
                (3,40),
                (15,40),
                (15,30),
                (10,30),
                (10,10),
                (15,10)
            ]
        ],
        'W': [
            [ (0, 0),(  0, 45),
                (5, 45),
                (8, 25
                ),(12, 25
                ),(15, 45
                ),(20, 45
                ),(20, 0
                ),(16.199999999999996, 0.3
                ),(15.099999999999998, 15.100000000000014
                ),(10.199999999999994, 9.999999999999995
                ),(6.500000000000001, 14.800000000000004),(
                4.4, 0.09999999999999731)
            ] ],
        'E': [
            [(0,0),(0,40),(15,40),(15,32),(5,32),(5,22),(12,22),(12,18),
             (5,18),(5,8),(15,8),(15,0)]
        ],
        'D': [
            [(0,0),(0,40),(11,40),(17,34),(17,6),(11,0)]
        ],
        'G': [
            [
                (5,0),(15,0),(20,5),(20,12),
                (15,12),(15,9),(12,9),(12,15),(15,15),(15,18),(20,18),(20,33),
                (15,40),(5,40),(0,33),(0,5)
            ]
        ],
        'R': [
            [
                (0,45),(0,0),(13,0),(18,5),(18,15),(13,20),(4,20),(4,25),(9,25),(20,45),
                (15,45),(10,35),(5,35)
            ]
        ],
        'O': [
            [(4,0),(11,0),(17,6),(17,34),(11,40),(4,40),(0,34),(0,6)]
        ],
        'U': [
            [(0,0),(0,35),(5,40),(12,40),(17,35),(17,0)]
        ],
        ' ': [],
        'P': [
            [
                (0,40),
                (0,0),
                (10,0),
                (15,5),
                (15,15),
                (10,20),
                (0,20)
            ]
        ],
        'S': [
            [(15,0),(5,0),(0,5),(0,15),(5,20),(10,20),(15,25),(15,35),
             (10,40),(0,40)]
        ],
        'T': [
            [(0,0),(15,0),(15,10),(10,10),(10,40),(5,40),(5,10),(0,10)]
        ],
        'A': [
            [(0,40),(5,0),(10,0),(15,40),(10,35),(5,35)]
        ]
    }

    class Star:
        """
        Inner class for parallax + twinkle star
        """
        def __init__(self, layer=0):
            self.x = random.randrange(0, WIDTH)
            self.y = random.randrange(0, HEIGHT)
            self.layer = layer
            self.speed = 0.5 + layer*0.5
            self.base_brightness = random.randint(100, 255)
            self.brightness = self.base_brightness
            self.color = (self.brightness,)*3
            self.twinkle_timer = random.randrange(0, 100)

        def update(self):
            self.y += self.speed
            if self.y > HEIGHT:
                self.y = 0
                self.x = random.randrange(0, WIDTH)
            twinkle_speed = random.uniform(0.01, 0.05)
            self.twinkle_timer += 1
            raw_bright = self.base_brightness + 30*math.sin(self.twinkle_timer*twinkle_speed)
            raw_bright = max(0, min(255, raw_bright))
            self.brightness = int(raw_bright)
            self.color = (self.brightness,)*3

        def draw(self, surface):
            size = 1 + self.layer
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), size)

    def __init__(self):
        if MUSIC_ENABLED:
            pygame.mixer.music.load("assets/cq1.wav")  # Replace with your MP3 file path
            pygame.mixer.music.play(-1)  # -1 loops the music indefinitely        

        self.screen=pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("WEDGEROGUE + FLASHING PRESS SPACE with starfield & sparkles")
        self.clock=pygame.time.Clock()
        self.running=True

        # starfield
        self.stars=[]
        for layer in range(3):
            for _ in range(100):
                self.stars.append(self.Star(layer))

        # orbits
        self.outer_ring=150
        self.inner_ring=110

        # sparkles
        self.center_x = WIDTH//2
        self.center_y = HEIGHT//2
        self.sparkles = self.generate_sparkles(self.center_x, self.center_y, radius=140, count=20)

        # main / sub text
        self.main_text  = "WEDGEROGUE"
        self.sub_text   = "PRESS SPACE"
        self.main_scale = 2.5
        self.sub_scale  = 1.0

        # position them near center
        self.main_text_x = self.center_x - 220
        self.main_text_y = self.center_y - 60
        self.sub_text_x  = self.center_x - 100
        self.sub_text_y  = self.center_y + 60

        self.timer=0

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
                if os.environ.get('PYGBAG') is None:
                    pygame.quit()
                    sys.exit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                    if os.environ.get('PYGBAG') is None:
                        pygame.quit()
                        sys.exit()
                if e.key in (pygame.K_SPACE, pygame.K_RETURN):
                    self.running = False

    def update(self):
        self.timer += 1
        for s in self.stars:
            s.update()

    def draw(self):
        self.screen.fill((0,0,0))

        # starfield
        for s in self.stars:
            s.draw(self.screen)

        # orbits
        self.draw_orbit(self.screen, (self.center_x, self.center_y),
                        self.outer_ring, 3, self.COLOR_GOLD)
        self.draw_orbit(self.screen, (self.center_x, self.center_y),
                        self.inner_ring, 3, self.COLOR_DARKER)

        # sparkles
        self.draw_sparkles(self.screen, self.sparkles)

        # main text
        self.draw_blocky_text(self.screen, self.main_text,
                              self.main_text_x, self.main_text_y,
                              scale=self.main_scale)

        # blinking sub text
        blink_rate = 40
        if (self.timer // blink_rate) % 2 == 0:
            self.draw_blocky_text(self.screen, self.sub_text,
                                  self.sub_text_x, self.sub_text_y,
                                  scale=self.sub_scale)

        pygame.display.flip()

    @staticmethod
    def draw_orbit(surface, center, radius, thickness=3, color=(255,200,50), segments=180):
        """
        Approximates a circle by connecting line segments
        """
        cx, cy = center
        old_x = cx + radius
        old_y = cy
        step_angle = (2*math.pi)/segments
        for i in range(1, segments+1):
            angle = i*step_angle
            nx = cx + radius*math.cos(angle)
            ny = cy + radius*math.sin(angle)
            pygame.draw.line(surface, color, (old_x,old_y), (nx,ny), thickness)
            old_x, old_y = nx, ny

    @staticmethod
    def generate_sparkles(center_x, center_y, radius=120, count=20):
        sparkles=[]
        for _ in range(count):
            angle = random.uniform(0, 2*math.pi)
            dist  = random.uniform(20, radius)
            sx = center_x + dist*math.cos(angle)
            sy = center_y + dist*math.sin(angle)
            size = random.randint(1,3)
            sparkles.append((sx,sy,size))
        return sparkles

    @staticmethod
    def draw_sparkles(surface, sparkles):
        for (sx,sy,size) in sparkles:
            pygame.draw.line(surface, FinalFlashingPressStart.COLOR_STARS,
                             (sx-size, sy), (sx+size, sy), 1)
            pygame.draw.line(surface, FinalFlashingPressStart.COLOR_STARS,
                             (sx, sy-size), (sx, sy+size), 1)

    @classmethod
    def draw_filled_letter(cls, surface, polygons, ox, oy, scale,
                           fill_color=None, outline_color=None,
                           outline_width=3):
        if fill_color is None:
            fill_color = cls.COLOR_GOLD
        if outline_color is None:
            outline_color = cls.COLOR_DARKER
        for poly in polygons:
            pts=[]
            for(px,py) in poly:
                sx = ox+px*scale
                sy = oy+py*scale
                pts.append((sx,sy))
            pygame.draw.polygon(surface, fill_color, pts)
            pygame.draw.polygon(surface, outline_color, pts, width=outline_width)

    @classmethod
    def draw_blocky_text(cls, surface, text, x, y, scale=2.0,
                         fill_color=None, outline_color=None,
                         outline_width=3):
        """
        R/W => bigger => slightly more spacing
        E,D,G,O,U,P,S,T,A => normal spacing
        """
        if fill_color is None:
            fill_color=cls.COLOR_GOLD
        if outline_color is None:
            outline_color=cls.COLOR_DARKER

        normal_spacing=18
        wide_spacing=22
        cx=x
        for ch in text:
            if ch==' ':
                cx+=(normal_spacing+8)*scale
                continue
            up=ch.upper()
            polys=cls.LETTERS.get(up)
            if not polys:
                cx+=normal_spacing*scale
                continue
            # fill + outline
            cls.draw_filled_letter(surface, polys, cx, y, scale,
                                   fill_color, outline_color, outline_width)
            if up in ('W','R'):
                cx+=wide_spacing*scale
            else:
                cx+=normal_spacing*scale

class Particle:
    """
    A flexible particle that can draw either a circle or a star shape,
    based on a 'shape' param. It also fades out over time.
    """
    def __init__(self, pos, vel, color, shape='circle', lifetime=30, radius=3):
        self.pos = Vector2(pos)
        self.vel = Vector2(vel)
        self.color = color
        self.shape = shape
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.radius = radius
        self.timer = 0  # for rotation if shape=='star'

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.timer += 1

    def draw(self, surface):
        alpha_ratio = self.lifetime / self.max_lifetime
        c = (
            int(self.color[0] * alpha_ratio),
            int(self.color[1] * alpha_ratio),
            int(self.color[2] * alpha_ratio)
        )

        if self.shape == 'circle':
            # simple circle
            pygame.draw.circle(surface, c, (int(self.pos.x), int(self.pos.y)), self.radius)
        elif self.shape == 'star':
            # rotating star shape
            num_spikes = 5
            points = []
            angle = self.timer * 0.3  # rotation speed
            for i in range(num_spikes * 2):
                # outer vs inner radius
                r = self.radius if (i % 2 == 0) else (self.radius * 0.5)
                theta = angle + (math.pi * 2 * i) / (num_spikes * 2)
                x = self.pos.x + r * math.cos(theta)
                y = self.pos.y + r * math.sin(theta)
                points.append((x, y))
            pygame.draw.polygon(surface, c, points)

    def is_dead(self):
        return self.lifetime <= 0

class ParticleSystem:
    """
    A single system that handles both 'hit spark' particles (circles)
    and 'star-burst' fancy particles, all in one container.
    """
    def __init__(self):
        self.particles = []

    def spawn_hit_particles(self, pos, color=(255, 200, 50), count=8):
        """
        Spawns small 'circle' spark particles, typically on a hit/explosion.
        """
        for _ in range(count):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(2, 5)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            p = Particle(pos, (vx, vy), color, shape='circle', lifetime=20, radius=3)
            self.particles.append(p)

    def spawn_star_burst(self, pos, color=(255, 255, 0), count=8):
        """
        Spawns 'star' shaped particles bursting outward.
        """
        for _ in range(count):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(1.5, 4.0)
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            # shape='star' => rotate star polygon
            p = Particle(pos, (vx, vy), color, shape='star', lifetime=30, radius=6)
            self.particles.append(p)

    def update(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if not p.is_dead()]

    def draw(self, surface):
        for p in self.particles:
            p.draw(surface)

class Bullet:
    def __init__(self, pos, vel, color=(255,255,255), homing=0, bigger=False, exploding=False):
        self.pos = Vector2(pos)
        self.vel = Vector2(vel)
        self.color = color
        self.life = 120
        self.homing = homing
        self.bigger = bigger
        self.exploding = exploding
        self.radius = 6 if bigger else 3

    def update(self, enemies):
        if self.homing > 0 and enemies:
            nearest_enemy = min(enemies, key=lambda e: (e.pos - self.pos).length_squared())
            diff = nearest_enemy.pos - self.pos
            dist = diff.length()
            if dist>0:
                diff.normalize_ip()
                steer = 0.05 if self.homing == 1 else 0.1
                self.vel = self.vel.lerp(diff * self.vel.length(), steer)
        self.pos += self.vel
        self.life-=1

    def draw(self, surface):
        pygame.draw.circle(surface,self.color,(int(self.pos.x),int(self.pos.y)), self.radius)

    def is_dead(self):
        return self.life<=0


class HUDSystem:
    """
    A stylized, 'sci-fi' HUD with extracted constants for easier tweaking.
    """

    # ------------------------ STYLE CONSTANTS ------------------------
    # Font
    SCIFI_FONT_PATH: str = None    # e.g. "Orbitron-Regular.ttf" or None for fallback
    FONT_SIZE: int = 18            # base font size
    SMALL_FONT_OFFSET: int = 6     # small font is (FONT_SIZE - SMALL_FONT_OFFSET)

    # Glow & Animation
    GLOW_PHASE_SPEED: float = 0.002   # multiplier for dt to animate glow
    GLOW_ALPHA_VALUES = [50, 35, 20]  # from outer to inner glow

    # Colors
    COLOR_TEXT = (255, 255, 255)
    COLOR_HP = (255, 30, 30)
    COLOR_SHIELD = (30, 200, 255)
    COLOR_BOMBS = (255, 220, 0)
    COLOR_COOL = (255, 150, 0)
    COLOR_SCORE = (150, 255, 150)
    COLOR_BAR_BG = (40, 40, 40)       # behind gradient
    COLOR_OUTLINE = (80, 80, 80)
    COLOR_SHADOW = (0, 0, 0, 100)     # partial alpha shadow

    # Sizing & Layout
    BAR_WIDTH = 200
    BAR_HEIGHT = 14
    PADDING = 28

    def __init__(self):
        """
        Loads fonts & initializes internal states.
        """
        # Attempt to load a sci-fi TTF (like 'Orbitron'), fallback if not found
        if self.SCIFI_FONT_PATH:
            self.font = pygame.font.Font(self.SCIFI_FONT_PATH, self.FONT_SIZE)
            self.small_font = pygame.font.Font(
                self.SCIFI_FONT_PATH,
                self.FONT_SIZE - self.SMALL_FONT_OFFSET
            )
        else:
            # fallback to a system font
            self.font = pygame.font.SysFont("bahnschrift", self.FONT_SIZE, bold=True)
            self.small_font = pygame.font.SysFont("bahnschrift",
                                                  self.FONT_SIZE - self.SMALL_FONT_OFFSET,
                                                  bold=True)

        self.glow_phase = 0.0  # for text glow animation

    def update(self, dt: float):
        """
        Called each frame with dt (milliseconds or so) to animate glow or other effects.
        """
        # animate glow_phase from 0..2*pi
        self.glow_phase += dt * self.GLOW_PHASE_SPEED
        if self.glow_phase > math.pi * 2:
            self.glow_phase -= math.pi * 2

    def draw(self, surface, player, screen_w, screen_h):
        """
        Draw the entire HUD in corners:
         - Top-left => HP & Shield
         - Top-right => Score
         - Bottom-left => bombs
         - Bottom-right => special cooldown (if present)
        """

        # 1) HP & Shield top-left
        tl_x = self.PADDING
        tl_y = self.PADDING
        self.draw_bar_with_label(
            surface, label="HP",
            current=player.hp, maximum=player.max_hp,
            pos=(tl_x, tl_y), color=self.COLOR_HP
        )
        tl_y += self.BAR_HEIGHT + self.PADDING

        if getattr(player, "shield_max", 0) > 0:
            self.draw_bar_with_label(
                surface, label="SHIELD",
                current=player.shield_hp, maximum=player.shield_max,
                pos=(tl_x, tl_y), color=self.COLOR_SHIELD
            )
            tl_y += self.BAR_HEIGHT + self.PADDING

        # 2) Score top-right
        score_label = f"SCORE {player.score}"
        self.draw_glow_text(surface, score_label, self.font, self.COLOR_SCORE,
                            x=screen_w - self.PADDING, y=self.PADDING,
                            align="top-right")

        # 3) Bombs bottom-left
        if "bombs" in getattr(player, "upgrades", {}):
            bombs_label = f"BOMBS {player.upgrades['bombs']}"
            self.draw_glow_text(surface, bombs_label, self.font, self.COLOR_BOMBS,
                                x=self.PADDING,
                                y=screen_h - self.PADDING,
                                align="bottom-left")

        # 4) Special cooldown bottom-right
        if "special_weapon_cooldown" in getattr(player, "upgrades", {}):
            val = player.upgrades["special_weapon_cooldown"]
            max_cool = 100
            bar_x = screen_w - (self.BAR_WIDTH + self.PADDING)
            bar_y = screen_h - (self.BAR_HEIGHT + self.PADDING)
            self.draw_bar(
                surface, current=val, maximum=max_cool,
                pos=(bar_x, bar_y),
                width=self.BAR_WIDTH, height=self.BAR_HEIGHT,
                fill_color=self.COLOR_COOL,
                label="SPECIAL", label_align="right"
            )

    def draw_bar_with_label(self, surface, label, current, maximum, pos, color):
        """
        Helper that draws a horizontal bar + label on top or above the bar.
        """
        self.draw_bar(
            surface,
            current=current,
            maximum=maximum,
            pos=pos,
            width=self.BAR_WIDTH,
            height=self.BAR_HEIGHT,
            fill_color=color,
            label=label,
            label_align="left"
        )

    def draw_bar(self, surface, current, maximum, pos,
                 width, height, fill_color,
                 label=None, label_align="left"):
        """
        A 'gradient fill' bar with a small outline and optional label.
        label_align: "left" or "right"
        """
        x, y = pos
        # Outline
        outline_rect = pygame.Rect(x - 1, y - 1, width + 2, height + 2)
        pygame.draw.rect(surface, self.COLOR_OUTLINE, outline_rect, 1)

        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, self.COLOR_BAR_BG, bg_rect)

        # fraction
        frac = float(current) / float(maximum) if maximum > 0 else 0.0
        frac = max(0.0, min(1.0, frac))
        fill_w = int(width * frac)

        # gradient fill
        if fill_w > 0:
            grad_surf = pygame.Surface((width, height))
            for row in range(height):
                if height > 1:
                    rf = row / float(height - 1)
                else:
                    rf = 0
                r = int(fill_color[0] * (1 - rf) + 20 * rf)
                g = int(fill_color[1] * (1 - rf) + 20 * rf)
                b = int(fill_color[2] * (1 - rf) + 20 * rf)
                pygame.draw.line(grad_surf, (r, g, b), (0, row), (width, row))
            surface.blit(grad_surf, (x, y),
                         area=pygame.Rect(0, 0, fill_w, height))

        # numeric text
        val_text = f"{int(current)}/{int(maximum)}"
        text_surf = self.small_font.render(val_text, True, self.COLOR_TEXT)
        cx = x + width // 2 - text_surf.get_width() // 2
        cy = y + height // 2 - text_surf.get_height() // 2
        # small drop shadow
        surface.blit(text_surf, (cx + 1, cy + 1))
        surface.blit(text_surf, (cx, cy))

        # label
        if label:
            label_surf = self.small_font.render(label, True, self.COLOR_TEXT)
            if label_align == "left":
                lx = x
            else:
                lx = x + width - label_surf.get_width()
            ly = y - label_surf.get_height() - 2
            # label drop shadow
            surface.blit(label_surf, (lx + 1, ly + 1))
            surface.blit(label_surf, (lx, ly))

    def draw_glow_text(self, surface, text, font, color, x, y, align="top-left"):
        """
        Renders text with a 'glow' effect by drawing multiple times with partial alpha.
        'align' can be: 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
        """
        text_upper = text.upper()
        # Base text surf
        base_surf = font.render(text_upper, True, color)
        rect = base_surf.get_rect()

        # alignment
        if "right" in align:
            rect.right = x
        else:
            rect.left = x
        if "bottom" in align:
            rect.bottom = y
        else:
            rect.top = y

        # create a temporary surface
        glow_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        glow_surf.fill((0, 0, 0, 0))

        # We'll gently vary the glow radius with self.glow_phase if we want
        glow_radius = 2 + 2 * math.sin(self.glow_phase)

        # For each alpha layer, do:
        for alpha_val in self.GLOW_ALPHA_VALUES:
            # 1) Render in white first
            temp_text = font.render(text_upper, True, (255, 255, 255, 255))
            temp_text = temp_text.convert_alpha()
            # 2) Set partial alpha
            temp_text.set_alpha(alpha_val)
            # 3) Colorize using multiply
            color_surf = pygame.Surface(temp_text.get_size(), pygame.SRCALPHA)
            color_surf.fill((color[0], color[1], color[2], 255))
            temp_text.blit(color_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            # 4) ring offsets
            for dx, dy in [
                (-glow_radius, 0), (glow_radius, 0),
                (0, -glow_radius), (0, glow_radius)
            ]:
                gx = glow_surf.get_width() // 2 - temp_text.get_width() // 2 + int(dx)
                gy = glow_surf.get_height() // 2 - temp_text.get_height() // 2 + int(dy)
                glow_surf.blit(temp_text, (gx, gy))

        # finally, topmost text fully opaque
        center_x = glow_surf.get_width() // 2 - base_surf.get_width() // 2
        center_y = glow_surf.get_height() // 2 - base_surf.get_height() // 2
        glow_surf.blit(base_surf, (center_x, center_y))

        # position glow_surf onto the main surface
        final_rect = glow_surf.get_rect()
        if "right" in align:
            final_rect.right = rect.right
        else:
            final_rect.left = rect.left
        if "bottom" in align:
            final_rect.bottom = rect.bottom
        else:
            final_rect.top = rect.top

        surface.blit(glow_surf, final_rect)


class FancyLootToken:
    """
    A 'fancy' loot token that drifts with a wave motion,
    draws as a rotating star polygon, and on collision,
    spawns star particles in a separate StarParticleSystem
    and can be removed.
    """
    def __init__(self, pos, color, value=10, particle_system=None):
        self.pos = Vector2(pos)
        self.vel = Vector2(random.uniform(-0.5,0.5), random.uniform(-0.5,0.5))
        self.color = color
        self.value = value
        self.radius = 8
        self.timer = 0
        # We keep a reference to a StarParticleSystem to spawn an explosion
        self.particle_system = particle_system

    def update(self):
        self.timer += 1

        # Gentle wave in velocity
        wave_mag = 0.2
        self.vel.y += wave_mag * math.sin(self.timer * 0.05)
        self.vel.x += wave_mag * math.cos(self.timer * 0.07)

        # Move & friction
        self.pos += self.vel * 0.3
        self.vel *= 0.98

        # (Optionally) wrap around the screen:
        # self.pos = wrap_position(self.pos, WIDTH, HEIGHT)

    def draw(self, surface):
        # rotating star polygon
        angle = self.timer * 0.1
        num_spikes = 5
        points = []
        for i in range(num_spikes * 2):
            r = self.radius if i % 2 == 0 else self.radius * 0.5
            theta = angle + (math.pi * 2 * i) / (num_spikes * 2)
            x = self.pos.x + r * math.cos(theta)
            y = self.pos.y + r * math.sin(theta)
            points.append((x,y))

        pygame.draw.polygon(surface, self.color, points)

    def check_collision(self, player):
        dist_sq = (self.pos - player.pos).length_squared()
        # Suppose player's bounding radius is ~20
        return dist_sq < (self.radius + 20)**2

    def on_collected(self):
        """
        Called when the loot is collected by the player.
        Spawns an 'explosion' of star particles if we have a system reference.
        """
        if self.particle_system:  # reference to your shared system
            self.particle_system.spawn_star_burst(self.pos, color=self.color, count=10)
        # You can also do a sound effect, etc.

class LootToken:
    def __init__(self, pos, color, value=10):
        self.pos=Vector2(pos)
        self.vel=Vector2(random.uniform(-1,1), random.uniform(-1,1))
        self.color=color
        self.value=value
        self.radius=8

    def update(self):
        self.pos+= self.vel*0.3
        self.vel*=0.98
        self.pos=wrap_position(self.pos, WIDTH, HEIGHT)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x),int(self.pos.y)), self.radius)

    def check_collision(self, player):
        dist_sq=(self.pos - player.pos).length_squared()
        return dist_sq < (self.radius+20)**2


class TrailSegment:
    def __init__(self, pos, angle, color, lifetime=30):
        self.pos = Vector2(pos)
        self.angle = angle
        self.color = color
        self.lifetime= lifetime

    def update(self):
        self.lifetime-=1

    def draw(self, surface):
        alpha_ratio=self.lifetime/30
        c=(int(self.color[0]*alpha_ratio),
           int(self.color[1]*alpha_ratio),
           int(self.color[2]*alpha_ratio))
        sp=[Vector2(15,0),Vector2(-8,8),Vector2(-8,-8)]
        rad=math.radians(self.angle)
        ca=math.cos(rad)
        sa=math.sin(rad)
        pts=[]
        for p in sp:
            x=p.x*ca - p.y*sa
            y=p.x*sa + p.y*ca
            pts.append((x+self.pos.x,y+self.pos.y))
        pygame.draw.polygon(surface,c,pts)


class PlayerShip:
    def __init__(self):
        self.pos = Vector2(WIDTH // 2, HEIGHT // 2)
        self.vel = Vector2(0, 0)
        self.angle = -90
        self.color = (255, 255, 255)
        self.bullets = []
        self.trails = []
        self.cooldown = 0
        self.score = 0
        self.engine_sound = 0

        # HP
        self.max_hp = 50
        self.hp = 50

        # Shield
        self.shield_hp = 0
        self.shield_max = 0

        # Upgrades
        self.upgrades = {
            "homing_bullet_level": 0,
            "bigger_bullets": False,
            "exploding_bullets": False,
            "shield": False,
            "teleporter": False,
            "short_range_autofire": False,
            "mines": False,
            "homing_mines": False,
        }

        # Start with alt mode on (press 'P' to switch to classic)
        self.alt_mode = True
        self.prev_p_pressed = False

        # Load and scale sprite
        self.sprite = pygame.image.load("assets/ship_01.png").convert_alpha()
        self.sprite_height = 40  # Match the height of the ship
        sprite_width = int(self.sprite.get_width() * (self.sprite_height / self.sprite.get_height()))
        self.sprite = pygame.transform.scale(self.sprite, (sprite_width, self.sprite_height))
        #self.sprite_alpha = 128  # Set default transparency (0-255)

    def draw(self, surface):
        # Draw trails
        for t in self.trails:
            t.draw(surface)

        # Draw wedge (polygon ship representation)
        sp = [Vector2(20, 0), Vector2(-10, 10), Vector2(-10, -10)]
        rad = math.radians(self.angle)
        ca = math.cos(rad)
        sa = math.sin(rad)
        pts = []
        for p in sp:
            x = p.x * ca - p.y * sa
            y = p.x * sa + p.y * ca
            pts.append((x + self.pos.x, y + self.pos.y))
        pygame.draw.polygon(surface, self.color, pts, width=2)

        # Draw bullets
        for b in self.bullets:
            b.draw(surface)

        # Draw the sprite
        rotated_sprite = pygame.transform.rotate(self.sprite, -self.angle)
        sprite_rect = rotated_sprite.get_rect(center=(self.pos.x, self.pos.y))
        #rotated_sprite.set_alpha(self.sprite_alpha)  # Apply transparency
        surface.blit(rotated_sprite, sprite_rect.topleft)

    def reset(self):
        self.pos = Vector2(WIDTH // 2, HEIGHT // 2)
        self.vel = Vector2(0, 0)
        self.angle = -90
        self.bullets.clear()
        self.trails.clear()
        self.cooldown = 0
        self.engine_sound = 0
        # self.hp = self.max_hp  # if you want to restore HP here

    def update(self, keys, enemies):
        # Toggle with 'P'
        p_pressed = keys[pygame.K_p]
        if p_pressed and not self.prev_p_pressed:
            self.alt_mode = not self.alt_mode
            print(f"Player control mode => {'ALT' if self.alt_mode else 'CLASSIC'}")
        self.prev_p_pressed = p_pressed

        if not self.alt_mode:
            # CLASSIC MODE: rotation + thrust
            if keys[pygame.K_LEFT]:
                self.angle -= ROTATION_SPEED
            if keys[pygame.K_RIGHT]:
                self.angle += ROTATION_SPEED
            if keys[pygame.K_UP]:
                rad = math.radians(self.angle)
                force = Vector2(math.cos(rad), math.sin(rad)) * THRUST
                self.vel += force
                if self.vel.length() > MAX_SPEED:
                    self.vel.scale_to_length(MAX_SPEED)
            if keys[pygame.K_DOWN]:
                rad = math.radians(self.angle)
                force = Vector2(math.cos(rad), math.sin(rad)) * (-THRUST * 0.5)
                self.vel += force
                if self.vel.length() > MAX_SPEED:
                    self.vel.scale_to_length(MAX_SPEED)

        else:
            # ALT MODE (absolute direction) with some momentum
            accel = Vector2(0,0)
            accel_amount = 3  # how strongly we accelerate
            friction = 0.80     # mild friction => some momentum remains
            rate = 4
            if keys[pygame.K_LEFT]:
                accel.x -= accel_amount
                self.engine_sound = (self.engine_sound + 1) % rate
                if self.engine_sound == 0:
                    sfx_system.play_sound(SFX_MOVE_LEFT)
            if keys[pygame.K_RIGHT]:
                accel.x += accel_amount
                self.engine_sound = (self.engine_sound + 1) % rate
                if self.engine_sound == 0:
                    sfx_system.play_sound(SFX_MOVE_RIGHT)
            if keys[pygame.K_UP]:
                accel.y -= accel_amount
                self.engine_sound = (self.engine_sound + 1) % rate
                if self.engine_sound == 0:
                    sfx_system.play_sound(SFX_MOVE_UP)
            if keys[pygame.K_DOWN]:
                self.engine_sound = (self.engine_sound + 1) % rate
                if self.engine_sound == 0:
                    sfx_system.play_sound(SFX_MOVE_DOWN)
                accel.y += accel_amount

            # apply acceleration
            self.vel += accel
            # clamp to max speed
            if self.vel.length() > MAX_SPEED:
                self.vel.scale_to_length(MAX_SPEED)

            # if no input, friction
            if accel.length_squared() < 1e-6:
                self.vel *= friction

            # instant turn if speed > small threshold
            speed = self.vel.length()
            if speed > 0.1:
                self.angle = math.degrees(math.atan2(self.vel.y, self.vel.x))

        # Teleporter, mines, autofire
        if self.upgrades["teleporter"] and keys[pygame.K_t]:
            sfx_system.play_sound(SFX_TELEPORT)
            self.pos = Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT))

        if self.upgrades["mines"] and keys[pygame.K_m]:
            sfx_system.play_sound(SFX_MINE_DROP)
            self.drop_mine()

        if self.upgrades["short_range_autofire"]:
            self.auto_fire(enemies)

        # move & wrap
        self.pos += self.vel
        self.pos = wrap_position(self.pos, WIDTH, HEIGHT)

        # update bullets
        for b in self.bullets:
            b.update(enemies if b.homing > 0 else None)
        self.bullets = [b for b in self.bullets if not b.is_dead()]

        # update trails
        for t in self.trails:
            t.update()
        self.trails = [t for t in self.trails if t.lifetime > 0]

        # add new trail
        self.trails.append(TrailSegment(self.pos, self.angle, (0,255,0), 30))

        if self.cooldown > 0:
            self.cooldown -= 1

        # if hp <= 0 => handle
        if self.hp <= 0:
            print("Player destroyed!")
            # self.reset() # or game_state=END, etc.

    def shoot(self):
        if self.cooldown <= 0:
            rad = math.radians(self.angle)
            direction = Vector2(math.cos(rad), math.sin(rad))
            bullet_speed = 10
            bullet = Bullet(
                self.pos + direction*20,
                direction*bullet_speed,
                color=(0,255,0),
                homing=self.upgrades["homing_bullet_level"],
                bigger=self.upgrades["bigger_bullets"],
                exploding=self.upgrades["exploding_bullets"]
            )
            self.bullets.append(bullet)
            self.cooldown = 10

    def drop_mine(self):
        mine_vel = Vector2(0,0)
        if self.upgrades["homing_mines"]:
            mine_vel = Vector2(random.uniform(-1,1), random.uniform(-1,1))
        bullet = Bullet(
            self.pos,
            mine_vel,
            color=(255,0,0),
            homing=2 if self.upgrades["homing_mines"] else 0,
            bigger=True,
            exploding=True
        )
        bullet.life = 300
        self.bullets.append(bullet)

    def auto_fire(self, enemies):
        if enemies:
            for e in enemies:
                dist = (e.pos - self.pos).length()
                if dist < 200:
                    if self.cooldown <= 0:
                        diff = (e.pos - self.pos)
                        self.angle = math.degrees(math.atan2(diff.y, diff.x))
                        sfx_system.play_sound(SFX_AUTO_FIRE)
                        self.shoot()
                    break


class FancyCountdown:
    """
    A fancy countdown timer that can be started on demand.
    
    - Use `start(total_seconds)` to begin counting down.
    - Call `update()` each frame.
    - Call `draw(surface)` after your game draws, if `active` is True.
    - Each second from total_seconds..1 displays in the center,
      fading from transparent & big to opaque & normal size.
    - Once it finishes, `active=False`.

    Now revised so that when the second flips (from 3->2, etc.),
    the new second is *instantly* big & transparent in that same frame.
    """

    def __init__(self, fps=60, base_font_size=100, color=(255,255,255), font_name=None):
        self.fps = fps
        self.base_font_size = base_font_size
        self.color = color
        self.font_name = font_name

        self.active = False
        self.frames_left = 0
        self.total_seconds = 0
        self.seconds_left = 0
        self.last_seconds_left = -1

    def start(self, total_seconds=10):
        """Begins the countdown from total_seconds..1."""
        self.total_seconds = total_seconds
        self.frames_left = total_seconds * self.fps
        self.active = True
        self.seconds_left = total_seconds
        self.last_seconds_left = -1

    def stop(self):
        """Manually stop the countdown."""
        self.active = False
        self.frames_left = 0

    def update(self):
        """Decrement the timer by 1 frame each call; ends when done."""
        if not self.active:
            return

        self.frames_left -= 1
        if self.frames_left < 0:
            self.frames_left = 0
            self.seconds_left = 0
            self.active = False
        else:
            # compute how many FULL seconds remain
            self.seconds_left = (self.frames_left + (self.fps - 1)) // self.fps
            # e.g. 359 frames => 359//60=5 w/ remainder => 6 in typical "round up"
        if self.seconds_left != self.last_seconds_left:
            # new second, play a sound
            sfx_system.play_sound(SFX_COUNTDOWN)
            self.last_seconds_left = self.seconds_left

    def draw(self, surface):
        """ 
        If active, draws the current second with a fade/scale effect.
        Each second starts at big & transparent, ends at normal & opaque.
        """
        if not self.active:
            return

        # If the timer is beyond range, skip
        if self.seconds_left < 1 or self.seconds_left > self.total_seconds:
            return

        # frames_in_current_second = frames_left % fps
        # if == 0 => we literally just switched to a new second
        frame_in_current_second = self.frames_left % self.fps

        # The fraction of how far along we are into the current second
        # By default:
        fraction_in_second = 1.0 - (frame_in_current_second / float(self.fps))

        # If we *just* switched to the new second, that means frame_in_current_second==0
        # we want fraction_in_second=0 => big & transparent
        if frame_in_current_second == 0:
            fraction_in_second = 0.0

        # alpha from 0..255
        alpha = int(255 * fraction_in_second)
        # scale from 2.0..1.0
        scale = 2.0 - fraction_in_second

        # dynamic font size
        import pygame
        font_size = int(self.base_font_size * scale)
        if font_size < 10:
            font_size = 10  # clamp

        font_obj = pygame.font.SysFont(self.font_name, font_size)
        text_str = str(self.seconds_left)
        text_surf = font_obj.render(text_str, True, self.color)
        text_surf = text_surf.convert_alpha()
        text_surf.set_alpha(alpha)

        center_x = surface.get_width() // 2
        center_y = surface.get_height() // 2
        rect = text_surf.get_rect(center=(center_x, center_y))
        surface.blit(text_surf, rect)

class Room:
    def __init__(self, idx, is_boss=False):
        self.idx = idx
        self.is_boss=is_boss  # if true => spawn a single boss
        self.enemies = []
        if not is_boss:
            for _ in range(random.randint(1,3)):
                ex = random.randint(100, WIDTH-100)
                ey = random.randint(100, HEIGHT-100)
                self.enemies.append(Enemy((ex, ey)))
        else:
            # spawn boss only
            self.enemies.append(Enemy((WIDTH/4, HEIGHT/4), is_boss=True))

        self.loot = []

        # NEW: let's add a particle system
        self.particles = ParticleSystem()
        self.countdown = FancyCountdown(
            fps=60,
            base_font_size=120,
            color=(255,255,255),
            font_name=None  # or "Arial", etc.
        )

    def start_collection(self):
        self.countdown.start(COLLECTION_TIME)

    def update(self, player, global_frame):
        # 1) update enemies
        for e in self.enemies:
            e.update(global_frame, self.enemies)

        # 2) check bullet collisions with enemies
        for e in self.enemies[:]:
            for pb in player.bullets:
                if e.hit_by(pb):
                    e.health -= 1
                    sfx_system.play_sound(13)
                    if pb.exploding:
                        # small AOE
                        for oe in self.enemies:
                            if (oe.pos - pb.pos).length() < 40:
                                oe.health -= 1
                    pb.life = 0
                    if e.health <= 0:
                        if not e.dropped_loot:
                            e.dropped_loot = True
                            self.loot.append(FancyLootToken(e.pos, e.color, e.points, particle_system=self.particles))
                            player.score += e.points
                        self.enemies.remove(e)
                        break

        # 3) check bullet collisions with player (enemy bullets)
        self.check_enemy_bullets_hit_player(player)

        # 4) update loot
        for l in self.loot[:]:
            l.update()
            if l.check_collision(player):
                l.on_collected()
                sfx_system.play_sound(27)
                # changed to give l.value * 5
                player.score += l.value * 5
                self.loot.remove(l)

        # 5) update particle system
        self.particles.update()

        # 6) collection timer
        self.countdown.update()

    def check_enemy_bullets_hit_player(self, player):
        """If an enemy bullet hits the player, do damage & spawn particles."""
        for e in self.enemies:
            for b in e.bullets[:]:  # copy so we can remove safely
                if self.bullet_hits_player(b, player):
                    # do damage
                    if (player.shield_hp > 0):
                        player.shield_hp -= 1
                    else:
                        player.hp -= 1
                    sfx_system.play_sound(SFX_HIT_PLAYER)                    
                    # spawn some sparks
                    self.particles.spawn_hit_particles(b.pos, color=(255, 200, 50), count=8)
                    # remove bullet
                    e.bullets.remove(b)

    def bullet_hits_player(self, bullet, player):
        dist_sq = (bullet.pos - player.pos).length_squared()
        # assume player's bounding radius ~ 20
        radius_sum = bullet.radius + 20
        return dist_sq < (radius_sum * radius_sum)

    def draw(self, surface):
        for e in self.enemies:
            e.draw(surface)
        for l in self.loot:
            l.draw(surface)
        # draw the particle system
        self.particles.draw(surface)
        self.countdown.draw(surface)


    def is_enemy_cleared(self):
        return (len(self.enemies) == 0)

    def is_collection_done(self):
        return (self.countdown.active == False)


################################################################################
# DIALOG: 5 lines for pre, 5 sets for post
# Use UPPERCASE speaker names, no "Room1 Pre" debug
################################################################################
g_pre_conversations = [
    [
        "COMMANDER TULIP:\nPilot, remember: strategy wins battles, not brute force.",
        "GENERAL XANTAR:\nBrute force works just fine if you use enough of it.",
        "DAN THE SPACE HAMSTER:\nI brute-forced my way into the snack stash yesterday. No regrets."
    ],
    [
        "TROY THE RHYTHM GURU:\nJust dodge to the beat, and you’ll be fine.",
        "COMMANDER TULIP:\nThis isn’t a dance recital, Troy.",
        "DAN THE SPACE HAMSTER:\nBut if it were, we’d definitely win 'best moves.'",
        "AI ASSISTANT:\nDancing detected. Enemies unimpressed. Continuing attack."
    ],
    [
        "GENERAL XANTAR:\nA true warrior must never hesitate.",
        "DAN THE SPACE HAMSTER:\nUnless there’s a cheese plate nearby. Priorities!",
        "COMMANDER TULIP:\nHamster, I’m starting to wonder how you made it onto this ship."
    ],
    [
        "AI ASSISTANT:\nLaser avoidance training is recommended.",
        "TROY THE RHYTHM GURU:\nOr just spin in circles while screaming. That works too.",
        "GENERAL XANTAR:\nThat’s a terrible strategy.",
        "DAN THE SPACE HAMSTER:\nIt’s more of a lifestyle, really."
    ],
    [
        "COMMANDER TULIP:\nYour mission is critical. Don’t mess it up.",
        "DAN THE SPACE HAMSTER:\nWow, motivational speeches are really your thing, huh?",
        "GENERAL XANTAR:\nMotivation isn’t required. Victory is.",
        "TROY THE RHYTHM GURU:\nI vote for snacks and good vibes instead. Who’s with me?",
        "AI ASSISTANT:\nSnacks do not improve survival rates."
    ],
    [
        "GENERAL XANTAR:\nThe enemies ahead are relentless. Prepare for a fight!",
        "COMMANDER TULIP:\nOr prepare to dodge. Whichever keeps us alive longer.",
        "AI ASSISTANT:\nDodge efficiency is currently suboptimal. Suggest improvement.",
        "TROY THE RHYTHM GURU:\nHey, suboptimal is my middle name!"
    ],
    [
        "TROY THE RHYTHM GURU:\nLife is a rhythm, pilot. Feel it, dodge it, win it.",
        "COMMANDER TULIP:\nI feel like your advice is always 90% nonsense.",
        "DAN THE SPACE HAMSTER:\nYeah, but the other 10% is gold. Like, actual gold sometimes."
    ],
    [
        "COMMANDER TULIP:\nThis ship is a finely tuned machine. Treat it with care.",
        "DAN THE SPACE HAMSTER:\nI thought it was held together by duct tape.",
        "AI ASSISTANT:\nStatement confirmed. Duct tape integrity at 84%.",
        "GENERAL XANTAR:\nIt’s enough for glory!"
    ],
    [
        "AI ASSISTANT:\nEnemy probability: 100%. Survival probability: questionable.",
        "DAN THE SPACE HAMSTER:\nI’m questioning why we brought you along.",
        "COMMANDER TULIP:\nShe’s saved us more times than you have, Hamster.",
        "TROY THE RHYTHM GURU:\nLet’s all agree to save each other this time."
    ],
    [
        "GENERAL XANTAR:\nA true warrior doesn’t fear death.",
        "DAN THE SPACE HAMSTER:\nI do! I fear it a lot!",
        "COMMANDER TULIP:\nThen channel that fear into something productive.",
        "TROY THE RHYTHM GURU:\nLike dodging! Fear-based dodging is the best kind."
    ],
    [
        "COMMANDER TULIP:\nStay focused, pilot. This is no time for distractions.",
        "DAN THE SPACE HAMSTER:\nUnless the distraction is snacks. Then it’s fine, right?",
        "AI ASSISTANT:\nSnacks detected: zero. Distraction detected: 100%.",
        "TROY THE RHYTHM GURU:\nDistraction? I call it 'creative improvisation.'",
        "GENERAL XANTAR:\nI call it 'a quick way to get us all vaporized.'"
    ],
    [
        "AI ASSISTANT:\nEnemies are numerous and heavily armed. Suggest caution.",
        "GENERAL XANTAR:\nA true warrior fears no enemy.",
        "DAN THE SPACE HAMSTER:\nWhat about highly explosive enemies? Asking for a friend.",
        "TROY THE RHYTHM GURU:\nExplosions are just fireworks you didn’t plan for!"
    ],
    [
        "COMMANDER TULIP:\nPilot, you’re the best we’ve got. Don’t let us down.",
        "DAN THE SPACE HAMSTER:\nWait, I thought *I* was the best we’ve got?",
        "GENERAL XANTAR:\nHamster, you’re barely on the list.",
        "TROY THE RHYTHM GURU:\nI’m on the list, right? For best vibes?"
    ],
    [
        "GENERAL XANTAR:\nThis room will test your limits, pilot.",
        "AI ASSISTANT:\nLimits detected: many. Breaking point estimated at 73%.",
        "TROY THE RHYTHM GURU:\nThat’s fine. Limits are just suggestions anyway.",
        "DAN THE SPACE HAMSTER:\nI suggest snacks. Lots of snacks."
    ],
    [
        "COMMANDER TULIP:\nEvery mission is a chance to prove yourself.",
        "DAN THE SPACE HAMSTER:\nProve myself? I’m already great!",
        "GENERAL XANTAR:\nGreat at eating everything in sight.",
        "TROY THE RHYTHM GURU:\nTo be fair, that’s an impressive skill.",
        "AI ASSISTANT:\nFood consumption rate: concerning. Recommend rationing."
    ]
]
g_post_dialogs = [
    {
        'prompt': "COMMANDER TULIP:\nWell done, pilot. What's next?",
        'choices': [
            ("1) \"Onward to glory!\"", "Score +10"),
            ("2) \"Time for a snack break.\"", "Score +5")
        ],
        'effects': [
            {'score': 10},
            {'score': 5}
        ]
    },
    {
        'prompt': "GENERAL XANTAR:\nShall I prepare you for the next battle?",
        'choices': [
            ("1) \"Yes, full repairs!\"", "HP restored."),
            ("2) \"No, I’m ready as is.\"", "No changes.")
        ],
        'effects': [
            {'repair': True},
            {}
        ]
    }
]

class DialogSystem:
    def __init__(self):
        # Pre-dialog: Banter-style conversations
        self.pre_conversations = g_pre_conversations

        # Post-dialog: Branching choices
        self.post_dialogs = g_post_dialogs

        # Internal state
        self.used_pre_conversations = set()
        self.current_pre_conversation = []
        self.current_post_dialog = None
        self.state = 'idle'
        self.timer = 0
        self.line_timer = 0  # Timer to track line display time
        self.DIALOG_TIME_PER_LINE = 240  # Frames per line (~4 seconds at 60 FPS)
        self.current_line = 0
        self.branch_mode = False
        self.selected_choice = None

    def is_pre_active(self):
        """Check if the pre-dialog is currently active."""
        return self.state == 'pre'

    def is_post_active(self):
        """Check if the post-dialog is currently active."""
        return self.state == 'post'

    def start_post_dialog(self):
        """Start a post-dialog with branching choices."""
        if len(self.post_dialogs) > 0:
            self.current_post_dialog = random.choice(self.post_dialogs)
            self.branch_mode = True
            self.selected_choice = None
            self.state = 'post'
            self.timer = 0
        else:
            self.state = 'idle'  # No post-dialog available

    def start_random_pre_dialog(self):
        """Start a random pre-dialog conversation."""
        available_conversations = [
            i for i in range(len(self.pre_conversations)) if i not in self.used_pre_conversations
        ]
        if available_conversations:
            selected_index = random.choice(available_conversations)
            self.used_pre_conversations.add(selected_index)
            self.current_pre_conversation = self.pre_conversations[selected_index]
            self.current_line = 0
            self.timer = 0
            self.line_timer = 0
            self.state = 'pre'
        else:
            self.current_pre_conversation = []
            self.state = 'idle'

    def next_line(self):
        self.current_line += 1
        self.line_timer = 0
        if self.current_line >= len(self.current_pre_conversation):
            # Finished all lines
            self.state = 'idle'  # triggers is_pre_done() == True

    def update(self, keys, player):
        """Update the dialog system based on input and timer."""
        # Handle PRE lines
        if self.state == 'pre':
            self.line_timer += 1
            # If user presses ENTER or we've shown the line long enough:
            if keys[pygame.K_RETURN] or self.line_timer > self.DIALOG_TIME_PER_LINE:
                self.next_line()        
        elif self.state == 'post' and self.current_post_dialog:
            if self.branch_mode:
                if keys[pygame.K_1]:
                    self.selected_choice = 0
                    self.branch_mode = False
                    self.apply_choice_effect(0, player)  # Pass player here
                elif keys[pygame.K_2]:
                    self.selected_choice = 1
                    self.branch_mode = False
                    self.apply_choice_effect(1, player)  # Pass player here
            else:
                self.timer += 1
                if self.timer > self.DIALOG_TIME_PER_LINE:
                    self.state = 'idle'

    def apply_choice_effect(self, choice_idx, player=None):
        """Apply effects of the selected post-dialog choice."""
        if self.current_post_dialog and 'effects' in self.current_post_dialog:
            effects = self.current_post_dialog['effects'][choice_idx]
            if player:  # Ensure player is passed
                if 'score' in effects:
                    player.score += effects['score']
                if 'repair' in effects:
                    player.hp = player.max_hp

    def draw(self, surface):
        """Draw the current dialog."""
        if self.state == 'pre' and self.current_pre_conversation:
            # Draw the current line of the pre-dialog
            if self.current_line < len(self.current_pre_conversation):
                line = self.current_pre_conversation[self.current_line]
                draw_text(surface, line, WIDTH // 2, HEIGHT // 2 - 100, center=True)
        elif self.state == 'post' and self.current_post_dialog:
            # Draw the post-dialog prompt
            prompt = self.current_post_dialog['prompt']
            draw_text(surface, prompt, WIDTH // 2, HEIGHT // 2 - 120, center=True)

            if self.branch_mode:
                # Draw branching choices
                y = HEIGHT // 2 - 40
                for idx, (choice, _) in enumerate(self.current_post_dialog['choices']):
                    draw_text(surface, choice, WIDTH // 2, y, color=(0, 255, 255), center=True)
                    y += 40
            elif self.selected_choice is not None:
                # Draw the selected choice result
                result = self.current_post_dialog['choices'][self.selected_choice][1]
                draw_text(surface, result, WIDTH // 2, HEIGHT // 2 - 40, center=True)

    def is_pre_active(self):
        """Check if pre-dialog is active."""
        return self.state == 'pre'

    def is_pre_done(self):
        """Check if pre-dialog is done."""
        return self.state != 'pre'

    def is_post_active(self):
        """Check if post-dialog is active."""
        return self.state == 'post'

    def is_post_done(self):
        """Check if post-dialog is done."""
        return self.state != 'post'



################################################################################
# STORE
################################################################################
class StoreSystem:
    def __init__(self):
        self.active=False
        self.message=""
        self.timer=0
        self.prev_keys = None  # store old key states

        # We'll track “already purchased” items with a set of indices
        self.items_purchased = set()

    def open_store(self):
        self.active=True
        self.message="Store: Press [1-9] to buy power-ups, ESC/ENTER to exit"
        self.timer=0

    def close_store(self):
        self.active=False
        self.message=""

    def update(self, player, keys):
        """
        Updates the store system logic:
         - If we haven't initialized self.prev_keys yet, do so now by building a dictionary for relevant keys.
         - If the store is inactive, return immediately.
         - ESC/ENTER => close store
         - For keys '1'..'9', check for a new press (false -> true transition) and buy the item if so.
        """

        # 1) If prev_keys is None => first frame => create a dict from all relevant keys
        if self.prev_keys is None:
            self.prev_keys = {}
            # We'll track these keys so we can detect "newly pressed":
            relevant_keys = [
                pygame.K_ESCAPE, pygame.K_RETURN,
                pygame.K_1, pygame.K_2, pygame.K_3,
                pygame.K_4, pygame.K_5, pygame.K_6,
                pygame.K_7, pygame.K_8, pygame.K_9
            ]
            # Initialize the dictionary (False for all keys)
            for key_code in relevant_keys:
                self.prev_keys[key_code] = keys[key_code]
            return

        # 2) If store is not active, do nothing
        if not self.active:
            return

        self.timer += 1

        # 3) Check if user pressed ESC or ENTER => close store
        if keys[pygame.K_ESCAPE] or keys[pygame.K_RETURN]:
            self.close_store()
            return

        # 4) Check numeric keys 1..9 for a new press
        for i in range(1, 10):
            key_code = getattr(pygame, f'K_{i}')  # e.g. pygame.K_1, ..., pygame.K_9
            this_key = keys[key_code]
            prev_key = self.prev_keys[key_code]

            # If it was up last frame and is down now => new press
            if this_key and not prev_key:
                self.buy_item(player, i - 1)

        # 5) Update prev_keys dictionary for next frame
        #    (i.e., store the *current* state of each relevant key)
        for k in self.prev_keys.keys():
            self.prev_keys[k] = keys[k]

    def buy_item(self, player, index):
        if index < 0 or index >= len(POWERUPS):
            return
        # Check if we already purchased this item
        if index in self.items_purchased:
            # Just block further attempts
            # self.message = "You've already bought this item!"
            return

        item = POWERUPS[index]
        print(f"DEBUG: Player score={player.score}, item cost={item['cost']}")
        if player.score >= item["cost"]:
            sfx_system.play_sound(SFX_BUY_SUCCESS)
            # Subtract cost & apply item
            player.score-=item["cost"]
            self.apply_powerup(player,index)
            # Mark as purchased
            self.items_purchased.add(index)
            self.message=f"Bought {item['name']} for {item['cost']} pts!"
#        else:
#            self.message=f"Not enough points to buy {item['name']} (cost {item['cost']})."

    def apply_powerup(self,player,index):
        name=POWERUPS[index]["name"]
        if name=="Homing Bullets":
            player.upgrades["homing_bullet_level"] = max(player.upgrades["homing_bullet_level"],1)
        elif name=="Better Homing Bullets":
            player.upgrades["homing_bullet_level"]=2
        elif name=="Exploding Bullets":
            player.upgrades["exploding_bullets"]=True
        elif name=="Bigger Bullets":
            player.upgrades["bigger_bullets"]=True
        elif name=="Ship Shield":
            player.upgrades["shield"]=True
            player.shield_max=30
            player.shield_hp=player.shield_max
        elif name=="Ship Teleporter":
            player.upgrades["teleporter"]=True
        elif name=="Short-Range Autofire":
            player.upgrades["short_range_autofire"]=True
        elif name=="Mines":
            player.upgrades["mines"]=True
        elif name=="Homing Mines":
            player.upgrades["mines"]=True
            player.upgrades["homing_mines"]=True

    def draw(self,surface):
        if self.active:
            draw_text(surface,self.message, WIDTH//2, HEIGHT//2 - 200, center=True)
            y=HEIGHT//2 - 140

            # We’ll loop over each item
            for i, p in enumerate(POWERUPS):
                # If we've already purchased it, dim it or show in gray
                if i in self.items_purchased:
                    color=(120,120,120)  # “dim” color
                else:
                    color=(255,255,0)   # normal bright color

                line=f"{i+1}) {p['name']} ({p['cost']} pts) - {p['desc']}"
                draw_text(surface,line, WIDTH//2, y, color=color, font_size=22, center=True)
                y+=30

################################################################################
# STAR MAP
################################################################################
class StarMapSystem:
    def __init__(self, num_rooms):
        self.active = False
        self.num_rooms = num_rooms

        # Node positions
        self.node_positions = []
        for i in range(num_rooms):
            nx = random.randint(50, 450)
            ny = random.randint(50, 450)
            self.node_positions.append((nx, ny))

        # Edges
        self.edges = []
        unconnected = list(range(num_rooms))
        random.shuffle(unconnected)
        while len(unconnected) > 1:
            a = unconnected.pop()
            b = unconnected[-1]
            e = tuple(sorted([a, b]))
            if e not in self.edges:
                self.edges.append(e)
        for _ in range(num_rooms):
            a = random.randint(0, num_rooms - 1)
            b = random.randint(0, num_rooms - 1)
            if a != b:
                e = tuple(sorted([a, b]))
                if e not in self.edges:
                    self.edges.append(e)

        self.MAP_TIME = 240    # frames or so
        self.timer = 0
        self.active = False

        # Animation time for swirl, pulsing lines, reticle, etc.
        self.anim_time = 0.0

        # For the "reticle zoom" or traveling highlight
        # We'll animate from prev_node -> next_node
        self.prev_node = 0
        self.next_node = 0
        self.travel_alpha = 0.0  # goes from 0..1 over time

    def open_map(self, prev_node_idx, next_node_idx):
        self.active = True
        self.timer = 0
        self.anim_time = 0.0
        self.prev_node = prev_node_idx
        self.next_node = next_node_idx
        self.travel_alpha = 0.0
        sfx_system.play_sound(SFX_STARMAP_OPEN)

    def close_map(self):
        self.active = False
        sfx_system.play_sound(SFX_STARMAP_CLOSE)
 
    def update(self, dt=1/60):
        if not self.active:
            return
        self.timer += 1
        self.anim_time += dt

        # We'll close after MAP_TIME frames
        if self.timer > self.MAP_TIME:
            self.close_map()

        # Animate travel_alpha from 0 to 1 over the last ~1 second
        # so that near the end, we see a "travel" from prev->next
        # or we can do the entire MAP_TIME
        total_travel_frames = self.MAP_TIME * 0.5  # half the time to do the travel
        if self.timer > self.MAP_TIME - total_travel_frames:
            # how far are we into the travel window?
            travel_progress = (self.timer - (self.MAP_TIME - total_travel_frames)) / total_travel_frames
            self.travel_alpha = min(1.0, max(0.0, travel_progress))
        else:
            self.travel_alpha = 0.0

    def draw(self, surface):
        if not self.active:
            return

        # 1) Dark background fill with some gradient or swirl
        self.draw_dark_background(surface)

        # 2) A "beveled border" or curved rectangle
        self.draw_beveled_border(surface, 200, 100, 500, 500, corner_radius=30)

        # 3) Draw pulsing edges inside that rectangle
        self.draw_pulsing_edges(surface, 200, 100)

        # 4) Draw nodes with a flicker, plus reticle that zooms in
        self.draw_nodes_and_reticle(surface, 200, 100)

        # 5) Optional text label
        font = pygame.font.SysFont(None, 26)
        label = font.render("Star Map - Warping...", True, (255, 255, 255))
        rect = label.get_rect(center=(200 + 500 // 2, 100 + 30))
        surface.blit(label, rect)

    def draw_dark_background(self, surface):
        """
        Draws a simple dark, possibly slightly gradient, background over entire screen.
        """
        w, h = surface.get_size()
        # Simple radial or vertical gradient approach:
        # Let's do a vertical gradient from (10,10,20) at top to (0,0,0) at bottom
        for y in range(h):
            ratio = y / h
            r = int(10 * (1 - ratio))
            g = int(10 * (1 - ratio))
            b = int(20 * (1 - ratio))
            pygame.draw.line(surface, (r, g, b), (0, y), (w, y))

    def draw_beveled_border(self, surface, x, y, w, h, corner_radius=20):
        """
        Draw a beveled or rounded rectangle.
        """
        # We'll just do a "rounded rect" approach.
        rect_color = (40, 40, 80)
        border_color = (100, 100, 150)

        main_rect = pygame.Rect(x, y, w, h)
        # fill the main area
        pygame.draw.rect(surface, rect_color, main_rect, border_radius=corner_radius)
        # draw the border
        pygame.draw.rect(surface, border_color, main_rect, width=4, border_radius=corner_radius)

    def draw_pulsing_edges(self, surface, map_x, map_y):
        """
        Draw the edges that connect nodes. We'll pulse the line color over time.
        """
        for (a, b) in self.edges:
            x1, y1 = self.node_positions[a]
            x2, y2 = self.node_positions[b]

            # pulsation
            pulse = 0.5 + 0.5 * math.sin(self.anim_time * 2 + a + b)
            edge_color = (
                int(255 * pulse),
                int(120 * pulse),
                int(255 * (1 - pulse))
            )

            pygame.draw.line(
                surface,
                edge_color,
                (map_x + x1, map_y + y1),
                (map_x + x2, map_y + y2),
                2
            )

    def draw_nodes_and_reticle(self, surface, map_x, map_y):
        """
        Draw the nodes, each with flickering ring.
        Then animate a reticle from prev_node -> next_node based on travel_alpha.
        """
        for i, (nx, ny) in enumerate(self.node_positions):
            px = map_x + nx
            py = map_y + ny

            # choose color for node
            c = (255, 255, 255)
            # If it's the previous node
            if i == self.prev_node:
                c = (0, 255, 0)
            # If it's the next node
            elif i == self.next_node:
                c = (255, 165, 0)

            # draw node center
            pygame.draw.circle(surface, c, (px, py), 8)

            # flicker ring
            ring_radius = 12 + 4 * math.sin(self.anim_time * 4 + i)
            ring_color = (
                min(255, c[0] + 50),
                min(255, c[1] + 50),
                min(255, c[2] + 50)
            )
            pygame.draw.circle(surface, ring_color, (px, py), max(1, int(ring_radius)), width=1)

        # RETICLE ANIMATION:
        # We'll draw a circle that moves from prev_node -> next_node based on travel_alpha in [0..1].
        if 0 <= self.travel_alpha <= 1.0:
            (prevx, prevy) = self.node_positions[self.prev_node]
            (nextx, nexty) = self.node_positions[self.next_node]

            # lerp
            cx = prevx + (nextx - prevx) * self.travel_alpha
            cy = prevy + (nexty - prevy) * self.travel_alpha
            px = map_x + cx
            py = map_y + cy

            # reticle size grows or shrinks
            ret_size = 20 + 10 * math.sin(self.anim_time * 2)
            ret_color = (255, 255, 0)
            # draw crosshair lines
            pygame.draw.line(surface, ret_color, (px - ret_size, py), (px + ret_size, py), 2)
            pygame.draw.line(surface, ret_color, (px, py - ret_size), (px, py + ret_size), 2)

            # draw a circle too
            pygame.draw.circle(surface, ret_color, (int(px), int(py)), int(ret_size), width=1)

class Star:
    def __init__(self, layer, player_ship):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        self.size = random.uniform(1, 3)  # Random size
        self.brightness = random.randint(50, 255)  # Random brightness
        self.layer = layer  # Determines parallax effect
        self.speed_multiplier = 0.2 + (layer * 0.2)  # Layer determines speed
        self.color = (self.brightness,) * 3
        self.player_ship = player_ship

    def update(self):
        # Parallax scrolling based on player ship's velocity
        self.x -= self.player_ship.vel.x * self.speed_multiplier
        self.y -= self.player_ship.vel.y * self.speed_multiplier

        # Wrap around screen
        if self.x < 0:
            self.x += WIDTH
        elif self.x >= WIDTH:
            self.x -= WIDTH

        if self.y < 0:
            self.y += HEIGHT
        elif self.y >= HEIGHT:
            self.y -= HEIGHT

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))


class Starfield:
    def __init__(self, num_stars, player_ship):
        self.stars = []
        self.player_ship = player_ship

        # Create stars across multiple layers
        for layer in range(3):  # Three layers for parallax
            for _ in range(num_stars // 3):
                self.stars.append(Star(layer, player_ship))

    def update(self):
        for star in self.stars:
            star.update()

    def draw(self, surface):
        for star in self.stars:
            star.draw(surface)

################################################################################
# DRAW INSTRUCTIONS
################################################################################
def draw_instructions(surface, game_state, current_room):
    lines = []
    lines.append("Controls:")
    lines.append("  Arrow Keys = Move/Rotate")
    lines.append("  SPACE = Shoot")
    lines.append("  T = Teleport (if you have teleporter)")
    lines.append("  M = Drop Mines (if you have mines)")

    if game_state=='PRE_DIALOG':
        lines.append("Press ENTER to skip the story text.")
    elif game_state=='POST_DIALOG':
        lines.append("Choose a dialog option (1 or 2).")
    elif game_state=='STORE':
        lines.append("Press 1-9 to buy power-ups, ESC/ENTER to exit store.")
    elif game_state=='COMBAT' or game_state=='COLLECTION':
        if not current_room.is_enemy_cleared():
            lines.append("Defeat all enemies to proceed!")
        else:
            if current_room.countdown.active:
                sec_left=int(current_room.countdown.seconds_left)
                lines.append(f"Collect loot! You have {sec_left} seconds to gather points.")
            else:
                lines.append("Loot collection is done. Waiting for next phase.")

    x=10
    y=HEIGHT-150
    for l in lines:
        draw_text(surface,l,x,y,font_size=18)
        y+=20

################################################################################
# STATES
################################################################################
STATE_PRE_DIALOG  = 'PRE_DIALOG'
STATE_COMBAT      = 'COMBAT'
STATE_COLLECTION  = 'COLLECTION'
STATE_POST_DIALOG = 'POST_DIALOG'
STATE_STORE       = 'STORE'
STATE_STAR_MAP    = 'STAR_MAP'
STATE_END         = 'END'

##############################################################################
# OptionsMenu (toggle music / sfx)
##############################################################################
class OptionsMenu:
    def __init__(self):
        self.active=False
        self.timer=0

    def open(self):
        self.active=True
    def close(self):
        self.active=False

    def update(self, keys):
        # check key press for toggling
        if keys[pygame.K_m]:
            toggle_music()
        if keys[pygame.K_s]:
            toggle_sfx()
        # press ESC to close?
        if keys[pygame.K_ESCAPE]:
            self.close()

    def draw(self, surface):
        if not self.active: return
        overlay=pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
        overlay.fill((0,0,0,150))
        surface.blit(overlay,(0,0))

        lines=[
            "OPTIONS MENU",
            "Press [M] to toggle Music",
            "Press [S] to toggle SFX",
            "ESC to return"
        ]
        y=200
        for l in lines:
            draw_text(surface,l,WIDTH//2,y,center=True,font_size=30)
            y+=40

################################################################################
# MAIN
################################################################################
async def main():
    pygame.init()
    FinalFlashingPressStart().run()

    if MUSIC_ENABLED: # TODO: starting with music disabled means it can't be started.
        pygame.mixer.music.load("assets/cq1.wav")  # Replace with your MP3 file path
        pygame.mixer.music.play(-1)  # -1 loops the music indefinitely        

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("WedgeRogue with HP & Fancy Loot")
    clock = pygame.time.Clock()

    player = PlayerShip()
    starfield = Starfield(100, player)

    # create rooms: last room => is_boss=True
    # e.g. if we have 5 rooms => room idx=4 => boss
    rooms=[]
    for i in range(NUM_ROOMS):
        if i==NUM_ROOMS-1:
            rooms.append(Room(i, is_boss=True))
        else:
            rooms.append(Room(i, is_boss=False))
    dialog = DialogSystem()
    store  = StoreSystem()
    star_map = StarMapSystem(NUM_ROOMS)
    hud = HUDSystem()

    options_menu=OptionsMenu()

    current_room_idx = 0
    game_state = STATE_PRE_DIALOG
    dialog.start_random_pre_dialog()  # start wave 0's pre-dialog

    global_frame = 0
    running = True
    while running:
        dt = clock.tick(FPS)
        global_frame += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Only allow ESC to quit if store & star map not active
                if event.key == pygame.K_ESCAPE:
                    if game_state not in (STATE_STORE, STATE_STAR_MAP):
                        running = False
                elif event.key == pygame.K_SPACE:
                    if game_state in (STATE_COMBAT, STATE_COLLECTION):
                        sfx_system.play_sound(SFX_SHOOT)
                        player.shoot()
                elif event.key==pygame.K_o:
                    # press 'O' to open/close options
                    if not options_menu.active:
                        options_menu.open()
                    else:
                        options_menu.close()
        # If game_state == STATE_END, we skip updating the gameplay logic
        if game_state == STATE_END:
            # We just continue the loop and draw "Game Over"
            keys = pygame.key.get_pressed()
            # store might still be open, etc. But typically we do nothing
        else:
            keys = pygame.key.get_pressed()
            # star map logic
            star_map.update()
            # store logic
            store.update(player, keys)
            # update dialogs
            if dialog.is_pre_active():
                dialog.update(keys, player)
            elif dialog.is_post_active():
                dialog.update(keys, player)

            # State Machine
            if options_menu.active:
                options_menu.update(keys)
            elif game_state == STATE_PRE_DIALOG:
                if dialog.is_pre_done():
                    # reset player & go to COMBAT
                    game_state = STATE_COMBAT

            elif game_state == STATE_COMBAT:
                current_room = rooms[current_room_idx]
                starfield.update()
                player.update(keys, current_room.enemies)
                current_room.update(player, global_frame)

                # if player's hp <= 0 => game over
                if player.hp <= 0:
                    game_state = STATE_END

                # if enemies cleared => collection
                elif current_room.is_enemy_cleared():
                    if not current_room.countdown.active:
                        current_room.start_collection()
                    game_state = STATE_COLLECTION

            elif game_state == STATE_COLLECTION:
                current_room = rooms[current_room_idx]
                starfield.update()
                player.update(keys, current_room.enemies)
                current_room.update(player, global_frame)

                # check hp
                if player.hp <= 0:
                    game_state = STATE_END
                elif current_room.is_collection_done():
                    dialog.start_post_dialog()
                    game_state = STATE_POST_DIALOG

            elif game_state == STATE_POST_DIALOG:
                if dialog.is_post_done():
                    store.open_store()
                    game_state = STATE_STORE

            elif game_state == STATE_STORE:
                if not store.active:
                    prev_idx = current_room_idx
                    next_idx = (current_room_idx + 1) % NUM_ROOMS
                    star_map.open_map(prev_idx, next_idx)
                    game_state = STATE_STAR_MAP
                    player.reset()

            elif game_state == STATE_STAR_MAP:
                if not star_map.active:
                    current_room_idx += 1
                    if current_room_idx >= NUM_ROOMS:
                        current_room_idx = 0
                        # or game_state=STATE_END
                    dialog.start_random_pre_dialog()
                    game_state = STATE_PRE_DIALOG

            # no changes for STATE_END here

        # RENDER
        screen.fill((0,0,0))

        if game_state == STATE_STAR_MAP:
            # if star map is active, show it
            star_map.draw(screen)
        else:
            # normal gameplay or end
            starfield.draw(screen)
            cur_room = rooms[current_room_idx]
            cur_room.draw(screen)
            player.draw(screen)
            dialog.draw(screen)
            store.draw(screen)
            hud.draw(screen, player, WIDTH, HEIGHT)
            draw_instructions(screen, game_state, cur_room)

        # If game_state == STATE_END, show "Game Over"
        if game_state == STATE_END:
            draw_text(screen, "GAME OVER!", WIDTH//2, HEIGHT//2, color=(255,50,50), font_size=48, center=True)
            draw_text(screen, "Press ESC to Quit", WIDTH//2, HEIGHT//2 + 60, color=(255,255,255), font_size=32, center=True)


        pygame.display.flip()

    if os.environ.get('PYGBAG') is None:
        pygame.quit()
        sys.exit()

asyncio.run(main())
