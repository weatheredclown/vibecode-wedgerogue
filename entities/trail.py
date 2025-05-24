import pygame
from pygame.math import Vector2
import math

class TrailSegment:
    def __init__(self, pos, angle, color, lifetime=30):
        self.pos = Vector2(pos) 
        self.angle = angle
        self.color = color # Expected to be a tuple like (r, g, b)
        self.lifetime = lifetime
        self.max_lifetime = float(lifetime) # For accurate alpha calculation

    def update(self):
        self.lifetime -= 1

    def draw(self, surface):
        if self.lifetime <= 0:
            return

        alpha_ratio = self.lifetime / self.max_lifetime # Use max_lifetime for accuracy
        
        # Create a new color tuple with alpha.
        # Original color is assumed to be (r,g,b).
        # Pygame's draw functions don't typically use an alpha in the color tuple directly for polygons.
        # Alpha is usually handled by surface alpha or special blend flags if drawing to a surface with SRCALPHA.
        # For simplicity, we'll fade the color towards black (or transparent if drawn on a per-pixel alpha surface).
        # The original code implies fading the color components.
        try:
            current_color = (
                int(self.color[0] * alpha_ratio),
                int(self.color[1] * alpha_ratio),
                int(self.color[2] * alpha_ratio)
            )
        except (IndexError, TypeError):
            # Fallback if self.color is not a simple (r,g,b) tuple
            # This might happen if pygame.Color object is passed, though less likely given context
            current_color = self.color 


        # Define the shape of the trail segment.
        # These points define a small triangle shape, similar to a ship's exhaust.
        sp = [Vector2(15,0), Vector2(-8,8), Vector2(-8,-8)] # Relative points
        
        rad = math.radians(self.angle)
        ca = math.cos(rad)
        sa = math.sin(rad)
        
        pts = []
        for p_vec in sp:
            x_rotated = p_vec.x * ca - p_vec.y * sa
            y_rotated = p_vec.x * sa + p_vec.y * ca
            pts.append((x_rotated + self.pos.x, y_rotated + self.pos.y))
        
        # Draw the polygon. If current_color has an alpha component and the surface supports it,
        # it might blend. Otherwise, it's just the faded RGB.
        pygame.draw.polygon(surface, current_color, pts)

    def is_dead(self):
        return self.lifetime <= 0
