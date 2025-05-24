import pygame
import random
import math # Added for math.sin if twinkle effect is kept complex, or can be removed if simplified
from config import WIDTH, HEIGHT

class Star:
    """Represents a single star in the parallax scrolling starfield."""
    def __init__(self, layer, player_ship, width=WIDTH, height=HEIGHT):
        # Parameters:
        # layer: Determines depth (parallax effect) and size/speed.
        # player_ship: Reference to the player object for parallax calculation.
        # width, height: Screen dimensions for positioning.
        
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.layer = layer # Layer 0 is "closest", higher numbers are further away
        
        # Speed and size are determined by layer for parallax effect
        # Farthest stars (higher layer) move slower and are smaller
        self.speed_multiplier = 0.1 + (self.layer * 0.15) # Base speed + increment per layer
        self.size = max(1, 3 - self.layer) # Smaller for deeper layers, min size 1
        
        self.brightness = random.randint(70, 180) # Dimmer stars for less distraction
        self.color = (self.brightness, self.brightness, self.brightness)
        
        self.player_ship = player_ship # Reference to player for parallax

        # Twinkle effect (optional, can be simplified or removed if too much)
        self.twinkle_timer = random.uniform(0, 2 * math.pi) # Random start phase
        self.twinkle_speed = random.uniform(0.01, 0.03)
        self.base_brightness = self.brightness


    def update(self):
        """Updates the star's position based on player movement (parallax) and screen wrapping."""
        # Parallax scrolling: Move star opposite to player's velocity, scaled by layer
        # This requires player_ship to have a 'vel' attribute (Vector2)
        if self.player_ship and hasattr(self.player_ship, 'vel'):
            self.x -= self.player_ship.vel.x * self.speed_multiplier
            self.y -= self.player_ship.vel.y * self.speed_multiplier
        
        # Screen wrapping logic
        if self.x < 0: self.x += WIDTH
        elif self.x >= WIDTH: self.x -= WIDTH
        if self.y < 0: self.y += HEIGHT
        elif self.y >= HEIGHT: self.y -= HEIGHT

        # Optional: Twinkle effect for brightness variation
        self.twinkle_timer += self.twinkle_speed
        # Use a sine wave for smooth brightness oscillation
        brightness_variation = 30 * math.sin(self.twinkle_timer)
        current_brightness = self.base_brightness + brightness_variation
        self.brightness = int(max(40, min(200, current_brightness))) # Clamp brightness
        self.color = (self.brightness, self.brightness, self.brightness)


    def draw(self, surface):
        """Draws the star on the given surface."""
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.size)


class Starfield:
    """Manages a collection of stars to create a parallax scrolling starfield."""
    def __init__(self, num_stars, player_ship):
        # Parameters:
        # num_stars: Total number of stars to create in the starfield.
        # player_ship: Reference to the player object, passed to each Star.
        
        self.stars = []
        self.player_ship = player_ship # Store player reference

        # Distribute stars across different layers for parallax depth
        num_layers = 3 # Example: 3 layers of depth
        stars_per_layer = num_stars // num_layers
        
        for layer_index in range(num_layers):
            for _ in range(stars_per_layer):
                # Pass player_ship to each Star instance
                self.stars.append(Star(layer=layer_index, player_ship=self.player_ship))

    def update(self):
        """Updates all stars in the starfield."""
        for star in self.stars:
            star.update()

    def draw(self, surface):
        """Draws all stars in the starfield onto the given surface."""
        for star in self.stars:
            star.draw(surface)
