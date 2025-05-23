import pygame
from pygame.math import Vector2
import math
import random

class Particle:
    def __init__(self, pos, vel, color, shape='circle', lifetime=30, radius=3):
        self.pos = Vector2(pos)
        self.vel = Vector2(vel)
        self.color = color
        self.shape = shape
        self.lifetime = lifetime
        self.max_lifetime = float(lifetime) # Ensure max_lifetime is float for division
        self.radius = radius
        self.timer = 0  # for rotation if shape=='star'

    def update(self):
        self.pos += self.vel
        self.lifetime -= 1
        self.timer += 1

    def draw(self, surface):
        if self.lifetime <= 0: # Don't draw dead particles
            return
            
        alpha_ratio = self.lifetime / self.max_lifetime
        # Ensure color components are integers after multiplication
        try:
            # Assuming color is a tuple of (r, g, b)
            current_color_tuple = (
                int(self.color[0] * alpha_ratio),
                int(self.color[1] * alpha_ratio),
                int(self.color[2] * alpha_ratio)
            )
            # Create a new Color object for rendering, as trying to set alpha on a tuple will fail
            render_color = pygame.Color(*current_color_tuple)
            # Pygame Color objects don't have an explicit alpha for drawing basic shapes directly,
            # it's usually handled by the surface alpha.
            # If per-pixel alpha is needed for shapes, it's more complex (e.g., drawing to a separate surface).
            # For now, this will just use the faded color.
        except (IndexError, TypeError): 
             # Fallback if self.color is not a simple tuple (e.g., already a pygame.Color object)
             # This part is tricky as pygame.Color objects are immutable in some aspects regarding direct item assignment.
             # A common way is to create a new color or handle it based on how colors are defined.
             # Given the original code, it's likely colors are tuples.
             # If it's a pygame.Color object, using its own methods or creating a new one is better.
             # For simplicity, assuming tuple colors as per original drawing logic.
             # If alpha blending is essential and colors are just tuples, one might need to draw on a per-pixel alpha surface.
             # However, the original code just faded the RGB values.
            render_color = self.color # Fallback to original color if tuple manipulation fails


        if self.shape == 'circle':
            pygame.draw.circle(surface, render_color, (int(self.pos.x), int(self.pos.y)), self.radius)
        elif self.shape == 'star':
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
            pygame.draw.polygon(surface, render_color, points)

    def is_dead(self):
        return self.lifetime <= 0

class ParticleSystem:
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
            p = Particle(pos, Vector2(vx, vy), color, shape='circle', lifetime=20, radius=3)
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
            p = Particle(pos, Vector2(vx, vy), color, shape='star', lifetime=30, radius=6)
            self.particles.append(p)

    def update(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if not p.is_dead()]

    def draw(self, surface):
        for p in self.particles:
            p.draw(surface)
