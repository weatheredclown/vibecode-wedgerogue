import pygame
from pygame.math import Vector2
import math
# No direct import of 'Enemy' needed here to avoid circular dependency.
# The 'enemies' argument in update() will be a list of objects that have a .pos attribute.

class Bullet:
    def __init__(self, pos, vel, color=(255,255,255), homing=0, bigger=False, exploding=False):
        self.pos = Vector2(pos)
        self.vel = Vector2(vel)
        self.color = color
        self.life = 120  # Default lifetime in frames
        self.homing = homing  # 0: no homing, 1: basic, 2: advanced
        self.bigger = bigger
        self.exploding = exploding
        self.radius = 6 if bigger else 3

    def update(self, enemies=None): # enemies can be None if no homing target
        if self.homing > 0 and enemies:
            # Find the nearest enemy for homing
            # This assumes enemies in the list have a 'pos' attribute (Vector2)
            # and 'is_dead' or similar attribute if filtering is needed (though not used here for min)
            try:
                nearest_enemy = min(enemies, key=lambda e: (e.pos - self.pos).length_squared())
                diff = nearest_enemy.pos - self.pos
                dist = diff.length()
                if dist > 0:  # Avoid division by zero if bullet is already at enemy position
                    diff.normalize_ip()
                    # Adjust steering force based on homing level
                    steer_strength = 0.05 if self.homing == 1 else 0.1 # Example strengths
                    self.vel = self.vel.lerp(diff * self.vel.length(), steer_strength)
            except ValueError: # Handle case where enemies list might be empty
                pass 
        
        self.pos += self.vel
        self.life -= 1

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

    def is_dead(self):
        return self.life <= 0
