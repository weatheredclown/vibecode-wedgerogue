import pygame
from pygame.math import Vector2
import math
import random

from utils import wrap_position
# from .trail import TrailSegment # Dependency for a later step
from .bullet import Bullet # Use the actual Bullet class from .bullet
from sfx import sfx_system
from config import (WIDTH, HEIGHT, ROTATION_SPEED, THRUST, MAX_SPEED,
                    SFX_TELEPORT, SFX_MINE_DROP, SFX_AUTO_FIRE, 
                    SFX_MOVE_LEFT, SFX_MOVE_RIGHT, SFX_MOVE_UP, SFX_MOVE_DOWN)

# Placeholder for TrailSegment if entities/trail.py is not yet created
try:
    from .trail import TrailSegment
except ImportError:
    class TrailSegment:
        def __init__(self, pos, angle, color, lifetime=30):
            self.pos = pos; self.angle = angle; self.color = color; self.lifetime = lifetime
            print("Warning: Using placeholder TrailSegment class in player.py")
        def update(self): self.lifetime -=1
        def draw(self, surface): pass # Minimal draw

# Placeholder for Bullet removed, using direct import from .bullet


class PlayerShip:
    def __init__(self):
        self.pos = Vector2(WIDTH // 2, HEIGHT // 2)
        self.vel = Vector2(0, 0)
        self.angle = -90 # Pointing upwards
        self.color = (255, 255, 255)
        self.bullets = []
        self.trails = []
        self.cooldown = 0
        self.score = 0
        self.engine_sound = 0 # Counter for engine sound throttling

        # HP
        self.max_hp = 50
        self.hp = 50

        # Shield
        self.shield_hp = 0
        self.shield_max = 0 # Max shield HP, can be upgraded

        # Upgrades dictionary
        self.upgrades = {
            "homing_bullet_level": 0,  # 0: no homing, 1: basic, 2: better
            "bigger_bullets": False,
            "exploding_bullets": False,
            "shield": False, # Whether the player has a shield upgrade
            "teleporter": False,
            "short_range_autofire": False,
            "mines": False,
            "homing_mines": False,
        }
        
        # Start with alt mode on (press 'P' to switch to classic)
        self.alt_mode = True 
        self.prev_p_pressed = False # To detect single key press for 'P'

        # Load and scale sprite
        try:
            self.sprite = pygame.image.load("assets/ship_01.png").convert_alpha()
            self.sprite_height = 40  # Match the height of the ship
            sprite_width = int(self.sprite.get_width() * (self.sprite_height / self.sprite.get_height()))
            self.sprite = pygame.transform.scale(self.sprite, (sprite_width, self.sprite_height))
        except pygame.error as e:
            print(f"Warning: Could not load player sprite: {e}")
            self.sprite = None
        # self.sprite_alpha = 128  # Set default transparency (0-255)


    def draw(self, surface):
        # Draw trails
        for t in self.trails:
            t.draw(surface)

        # Draw bullets
        for b in self.bullets:
            b.draw(surface)

        # Draw the sprite
        if self.sprite:
            rotated_sprite = pygame.transform.rotate(self.sprite, -self.angle) # Pygame rotates counter-clockwise
            sprite_rect = rotated_sprite.get_rect(center=(self.pos.x, self.pos.y))
            surface.blit(rotated_sprite, sprite_rect.topleft)
        else: # Fallback if sprite failed to load
            sp = [Vector2(20, 0), Vector2(-10, 10), Vector2(-10, -10)]
            rad = math.radians(self.angle)
            ca, sa = math.cos(rad), math.sin(rad)
            pts = [(p.x*ca - p.y*sa + self.pos.x, p.x*sa + p.y*ca + self.pos.y) for p in sp]
            pygame.draw.polygon(surface, self.color, pts, width=2)


    def reset(self):
        self.pos = Vector2(WIDTH // 2, HEIGHT // 2)
        self.vel = Vector2(0, 0)
        self.angle = -90
        self.bullets.clear()
        self.trails.clear()
        self.cooldown = 0
        self.engine_sound = 0
        # self.hp = self.max_hp # Restore HP, or handle in game logic

    def update(self, keys, enemies): # 'enemies' needed for homing and autofire
        # Toggle control mode with 'P'
        p_pressed = keys[pygame.K_p]
        if p_pressed and not self.prev_p_pressed:
            self.alt_mode = not self.alt_mode
            print(f"Player control mode => {'ALT' if self.alt_mode else 'CLASSIC'}")
        self.prev_p_pressed = p_pressed

        if not self.alt_mode:
            # CLASSIC MODE: rotation + thrust
            if keys[pygame.K_LEFT]: self.angle -= ROTATION_SPEED
            if keys[pygame.K_RIGHT]: self.angle += ROTATION_SPEED
            if keys[pygame.K_UP]:
                rad = math.radians(self.angle)
                force = Vector2(math.cos(rad), math.sin(rad)) * THRUST
                self.vel += force
                if self.vel.length() > MAX_SPEED: self.vel.scale_to_length(MAX_SPEED)
            if keys[pygame.K_DOWN]: # Optional reverse thrust or brake
                rad = math.radians(self.angle)
                force = Vector2(math.cos(rad), math.sin(rad)) * (-THRUST * 0.5)
                self.vel += force
        else:
            # ALT MODE (absolute direction) with some momentum
            accel = Vector2(0,0); accel_amount = 0.3; friction = 0.95; 
            
            sound_throttle_rate = 4 
            
            if keys[pygame.K_LEFT]: 
                accel.x -= accel_amount
                if self.engine_sound % sound_throttle_rate == 0: sfx_system.play_sound(SFX_MOVE_LEFT)
            if keys[pygame.K_RIGHT]: 
                accel.x += accel_amount
                if self.engine_sound % sound_throttle_rate == 0: sfx_system.play_sound(SFX_MOVE_RIGHT)
            if keys[pygame.K_UP]:   
                accel.y -= accel_amount
                if self.engine_sound % sound_throttle_rate == 0: sfx_system.play_sound(SFX_MOVE_UP)
            if keys[pygame.K_DOWN]:  
                accel.y += accel_amount
                if self.engine_sound % sound_throttle_rate == 0: sfx_system.play_sound(SFX_MOVE_DOWN)

            if accel.length_squared() > 0:
                 self.engine_sound +=1
            else:
                 self.engine_sound =0


            self.vel += accel
            if self.vel.length() > MAX_SPEED: self.vel.scale_to_length(MAX_SPEED)
            if accel.length_squared() < 1e-6 : self.vel *= friction 

            if self.vel.length() > 0.1: 
                self.angle = math.degrees(math.atan2(self.vel.y, self.vel.x))

        if self.upgrades["teleporter"] and keys[pygame.K_t]: 
            sfx_system.play_sound(SFX_TELEPORT)
            self.pos = Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT))
        
        if self.upgrades["mines"] and keys[pygame.K_m] and self.cooldown <=0: 
            sfx_system.play_sound(SFX_MINE_DROP)
            self.drop_mine()

        if self.upgrades["short_range_autofire"]:
            self.auto_fire(enemies)

        self.pos += self.vel
        self.pos = wrap_position(self.pos, WIDTH, HEIGHT)

        for b in self.bullets:
            b.update(enemies if b.homing > 0 else None)
        self.bullets = [b for b in self.bullets if not b.is_dead()]

        for t in self.trails: t.update()
        self.trails = [t for t in self.trails if t.lifetime > 0]
        if self.vel.length_squared() > 0.1: 
            self.trails.append(TrailSegment(self.pos, self.angle, (0,255,0), 30))

        if self.cooldown > 0: self.cooldown -= 1
        if self.hp <= 0: print("Player destroyed!") 

    def shoot(self):
        if self.cooldown <= 0:
            rad = math.radians(self.angle)
            direction = Vector2(math.cos(rad), math.sin(rad))
            bullet_pos = self.pos + direction * 20 
            
            bullet_homing = self.upgrades.get("homing_bullet_level", 0)
            bullet_bigger = self.upgrades.get("bigger_bullets", False)
            bullet_exploding = self.upgrades.get("exploding_bullets", False)

            self.bullets.append(Bullet(
                bullet_pos, direction * 10, color=(0,255,0), 
                homing=bullet_homing, bigger=bullet_bigger, exploding=bullet_exploding
            ))
            self.cooldown = 10 

    def drop_mine(self):
        mine_homing = 2 if self.upgrades.get("homing_mines", False) else 0
        mine_vel = Vector2(random.uniform(-1,1), random.uniform(-1,1)) if mine_homing else Vector2(0,0)
        
        bullet = Bullet(
            self.pos, mine_vel, color=(255,0,0), 
            homing=mine_homing, bigger=True, exploding=True
        )
        bullet.life = 300 
        self.bullets.append(bullet)

    def auto_fire(self, enemies):
        if self.cooldown <= 0 and enemies:
            for e in enemies:
                dist = (e.pos - self.pos).length()
                if dist < 200: 
                    diff = (e.pos - self.pos)
                    self.angle = math.degrees(math.atan2(diff.y, diff.x)) 
                    sfx_system.play_sound(SFX_AUTO_FIRE) 
                    self.shoot() 
                    break
