import pygame
from pygame.math import Vector2
import math
import random

from utils import wrap_position
from config import WIDTH, HEIGHT
# Assuming entities.particles will exist, if not, ParticleSystem might be undefined
# For robustness, a try-except could be used here if entities.particles might not be available
# during certain stages of refactoring, but the subtask implies it should be resolvable.
try:
    from entities.particles import ParticleSystem
except ImportError:
    # Placeholder if ParticleSystem is not available, to allow LootToken to be defined
    # This should be removed once all modules are correctly in place.
    class ParticleSystem:
        def __init__(self): print("Warning: Using placeholder ParticleSystem in loot.py")
        def spawn_star_burst(self, pos, color, count): pass


class LootToken:
    def __init__(self, pos, color, value=10):
        self.pos = Vector2(pos)
        self.vel = Vector2(random.uniform(-1,1), random.uniform(-1,1)) # Slight random drift
        self.color = color
        self.value = value
        self.radius = 8

    def update(self):
        self.pos += self.vel * 0.3  # Move with some velocity
        self.vel *= 0.98  # Apply friction/drag
        # Wrap around screen edges
        self.pos = wrap_position(self.pos, WIDTH, HEIGHT)

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, (int(self.pos.x), int(self.pos.y)), self.radius)

    def check_collision(self, player): # player object is expected to have a .pos (Vector2)
        # Simple circle collision detection
        dist_sq = (self.pos - player.pos).length_squared()
        # Assuming player has a bounding radius of ~20 for collision
        return dist_sq < (self.radius + 20)**2


class FancyLootToken:
    def __init__(self, pos, color, value=10, particle_system=None):
        self.pos = Vector2(pos)
        # Initial velocity for a gentle drifting wave motion
        self.vel = Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        self.color = color
        self.value = value
        self.radius = 8  # Visual radius for drawing the star polygon
        self.timer = 0   # For animation (wave motion, rotation)
        self.particle_system = particle_system # Optional: For spawning particles on collection

    def update(self):
        self.timer += 1

        # Gentle wave motion for drifting effect
        wave_amplitude_x = 0.1 * math.cos(self.timer * 0.07) # Slower wave for x
        wave_amplitude_y = 0.2 * math.sin(self.timer * 0.05) # Faster wave for y
        self.vel.x = math.cos(self.timer * 0.07) * 0.3 + wave_amplitude_x
        self.vel.y = math.sin(self.timer * 0.05) * 0.3 + wave_amplitude_y
        
        self.pos += self.vel * 0.3  # Apply movement
        self.vel *= 0.97  # Apply some friction to slow down over time if not continuously waved

        # Optional: Wrap around screen (can be enabled if desired)
        # self.pos = wrap_position(self.pos, WIDTH, HEIGHT)


    def draw(self, surface):
        # Rotating star polygon for visual representation
        angle = self.timer * 0.1  # Rotation speed
        num_spikes = 5
        points = []
        for i in range(num_spikes * 2):
            # Alternate between outer and inner radius for spikes
            r = self.radius if i % 2 == 0 else self.radius * 0.5
            theta = angle + (math.pi * 2 * i) / (num_spikes * 2)
            x = self.pos.x + r * math.cos(theta)
            y = self.pos.y + r * math.sin(theta)
            points.append((x,y))
        pygame.draw.polygon(surface, self.color, points)

    def check_collision(self, player): # player object expected to have .pos (Vector2)
        dist_sq = (self.pos - player.pos).length_squared()
        # Collision radius can be slightly larger than visual radius for easier collection
        return dist_sq < (self.radius + 20)**2 # Assuming player radius ~20

    def on_collected(self):
        """Called when the loot is collected by the player."""
        if self.particle_system:
            self.particle_system.spawn_star_burst(self.pos, color=self.color, count=10)
        # Sound effect for collection can be played here or in the main game logic that calls this
        # e.g., sfx_system.play_sound(SFX_LOOT_COLLECTED) # Requires SFX_LOOT_COLLECTED to be defined/imported
