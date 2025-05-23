import math
import numpy as np
import pygame
from config import MUSIC_ENABLED, SFX_ENABLED

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
    # from math import pi,sin # math is imported at the top of the file

    num_samps = int(sr*du)
    t = np.linspace(0,du,num_samps,endpoint=False)

    # 1) pitch factor
    if pt=="dn":
        pitch_factor=np.linspace(1.0,0.2,num_samps)
    elif pt=="up":
        pitch_factor=np.linspace(0.3,1.2,num_samps)
    elif pt=="sw":
        pitch_factor=1.0+0.4*np.sin(2*math.pi*1.0*t)
    elif pt=="st":
        step_sz=int(0.3*sr)
        pitch_factor=np.ones(num_samps)
        for i in range(num_samps):
            st_id=i//step_sz
            pitch_factor[i]=1.0+0.1*st_id
    else: # default to 'dn'
        pitch_factor=np.linspace(1.0,0.2,num_samps)

    # 2) freq-based FM
    cphase=2*math.pi*(cf*pitch_factor)*t
    mphase=2*math.pi*(mf*pitch_factor)*t
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
    
    env_parts = []
    if a_samps > 0: env_parts.append(np.linspace(0,1,a_samps,endpoint=False))
    if d1_samps > 0: env_parts.append(np.linspace(1,0.6,d1_samps,endpoint=False))
    if d2_samps > 0: env_parts.append(np.linspace(0.6,0.3,d2_samps,endpoint=False))
    if r_samps > 0: env_parts.append(np.linspace(0.3,0,r_samps,endpoint=False))

    if not env_parts: # Handle case where all envelope segments are zero
        env = np.zeros(num_samps)
    else:
        env = np.concatenate(env_parts)

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
      sr=cfg.get('sr',44100), # allow sample rate override
      echo=cfg.get('ec',False),ed=cfg.get('ed',0.2),ef=cfg.get('ef',0.5),
      dist=cfg.get('dist',False),th=cfg.get('th',0.3),
      at=cfg.get('at',0.08),d1=cfg.get('d1',0.10),d2=cfg.get('d2',0.10),rl=cfg.get('rl',0.5),
      pt=cfg.get('pt','dn')
    )

##############################################################################
# 25 SFX Presets with longer durations & nuanced envelopes
##############################################################################
DESC_FAMILY = [
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
        # This check ensures pygame.mixer is only initialized if needed and if not already done.
        if (SFX_ENABLED or MUSIC_ENABLED) and not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1)
        
        self.sounds = []
        # Pre-generate each preset for quick playback
        if pygame.mixer.get_init(): # Only generate sounds if mixer is initialized
            for i, params in enumerate(SFX_PRESETS):
                if SFX_ENABLED:
                    raw_samples = generate_fm_sound_from_dict(params)
                    try:
                        snd = pygame.mixer.Sound(buffer=raw_samples.tobytes())
                        self.sounds.append(snd)
                    except pygame.error as e:
                        print(f"Error creating sound for preset {i}: {e}")
                        # Optionally, append a None or a placeholder to keep indices consistent
                        self.sounds.append(None) 
        else:
            if SFX_ENABLED or MUSIC_ENABLED:
                 print("Warning: SFX_ENABLED or MUSIC_ENABLED is True, but pygame.mixer could not be initialized.")


    def play_sound(self, index):
        # only play if SFX_ENABLED and mixer is initialized
        if not SFX_ENABLED or not pygame.mixer.get_init():
            return
        if 0 <= index < len(self.sounds):
            if self.sounds[index]: # Check if the sound object is valid
                self.sounds[index].play()
        else:
            print(f"Invalid sfx index {index}")

# It's generally better to initialize sfx_system once in main.py after pygame.init()
# sfx_system = SfxSystem() # This line will be removed from here and handled in main.py
