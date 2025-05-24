import pygame
import sys
import os
import math
import random
from config import MUSIC_ENABLED, WIDTH, HEIGHT, FPS

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
    COLOR_STARS  = (255, 200,  50) # Used for sparkles

    # Letter polygons dictionary (as provided in the original main.py)
    LETTERS = {
        'C': [[(15,0),(3,0),(0,3),(0,37),(3,40),(15,40),(15,30),(10,30),(10,10),(15,10)]],
        'W': [[ (0, 0),(  0, 45),(5, 45),(8, 25),(12, 25),(15, 45),(20, 45),(20, 0),(16.199999999999996, 0.3),(15.099999999999998, 15.100000000000014),(10.199999999999994, 9.999999999999995),(6.500000000000001, 14.800000000000004),(4.4, 0.09999999999999731)] ],
        'E': [[(0,0),(0,40),(15,40),(15,32),(5,32),(5,22),(12,22),(12,18),(5,18),(5,8),(15,8),(15,0)]],
        'D': [[(0,0),(0,40),(11,40),(17,34),(17,6),(11,0)]],
        'G': [[(5,0),(15,0),(20,5),(20,12),(15,12),(15,9),(12,9),(12,15),(15,15),(15,18),(20,18),(20,33),(15,40),(5,40),(0,33),(0,5)]],
        'R': [[(0,45),(0,0),(13,0),(18,5),(18,15),(13,20),(4,20),(4,25),(9,25),(20,45),(15,45),(10,35),(5,35)]],
        'O': [[(4,0),(11,0),(17,6),(17,34),(11,40),(4,40),(0,34),(0,6)]],
        'U': [[(0,0),(0,35),(5,40),(12,40),(17,35),(17,0)]],
        ' ': [], # Space character
        'P': [[(0,40),(0,0),(10,0),(15,5),(15,15),(10,20),(0,20)]],
        'S': [[(15,0),(5,0),(0,5),(0,15),(5,20),(10,20),(15,25),(15,35),(10,40),(0,40)]],
        'T': [[(0,0),(15,0),(15,10),(10,10),(10,40),(5,40),(5,10),(0,10)]],
        'A': [[(0,40),(5,0),(10,0),(15,40),(10,35),(5,35)]]
    }

    class Star: # Inner class for stars in the starfield
        def __init__(self, layer=0):
            self.x = random.randrange(0, WIDTH) 
            self.y = random.randrange(0, HEIGHT)
            self.layer = layer # Determines parallax speed
            self.speed = 0.5 + layer * 0.5 # Deeper layers move slower
            self.base_brightness = random.randint(100, 200) # Dimmer stars for less noise
            self.brightness = self.base_brightness
            self.color = (self.brightness,) * 3 # Grayscale color
            self.twinkle_timer = random.randrange(0, 100) # For twinkle effect timing

        def update(self):
            # Star movement (simple downward scroll for splash screen)
            self.y += self.speed
            if self.y > HEIGHT:
                self.y = 0
                self.x = random.randrange(0, WIDTH)
            
            # Twinkle effect
            twinkle_speed = random.uniform(0.01, 0.03) # Slower twinkle
            self.twinkle_timer += 1
            # Sine wave for smooth brightness transition
            raw_bright = self.base_brightness + 20 * math.sin(self.twinkle_timer * twinkle_speed)
            self.brightness = int(max(50, min(220, raw_bright))) # Clamp brightness
            self.color = (self.brightness,) * 3

        def draw(self, surface):
            size = 1 + self.layer # Stars in deeper layers are smaller
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), size)

    def __init__(self):
        if MUSIC_ENABLED:
            try:
                pygame.mixer.music.load("assets/cq1.wav") 
                pygame.mixer.music.play(-1) # Loop indefinitely
            except pygame.error as e:
                print(f"Warning: Could not load or play music for splash screen: {e}")      

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("WEDGEROGUE - Splash Screen")
        self.clock = pygame.time.Clock()
        self.running = True

        self.stars = [self.Star(layer) for layer in range(3) for _ in range(80)] # Slightly fewer stars

        self.outer_ring_radius = 150
        self.inner_ring_radius = 110

        self.center_x = WIDTH // 2
        self.center_y = HEIGHT // 2
        self.sparkles = self.generate_sparkles(self.center_x, self.center_y, radius=140, count=25)

        self.main_text = "WEDGEROGUE"
        self.sub_text = "PRESS SPACE"
        self.main_scale = 2.8 # Slightly larger
        self.sub_scale = 1.2  # Slightly larger

        # Centering text based on its approximate width (can be pre-calculated or estimated)
        # This is a rough estimate; for perfect centering, render text first then get width.
        self.main_text_x = self.center_x - (len(self.main_text) * 18 * self.main_scale / 2) # Approx char width 18
        self.main_text_y = self.center_y - 70 # Adjusted Y position
        self.sub_text_x = self.center_x - (len(self.sub_text) * 18 * self.sub_scale / 2.5) # Approx char width 18
        self.sub_text_y = self.center_y + 70  # Adjusted Y position

        self.timer = 0 # General timer for animations

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()
        # Potentially stop music here if it shouldn't continue into the main game
        # pygame.mixer.music.stop()


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                if os.environ.get('PYGBAG') is None: # Check if running in Pygbag environment
                    pygame.quit()
                    sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    if os.environ.get('PYGBAG') is None:
                        pygame.quit()
                        sys.exit()
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    self.running = False # Exit splash screen to start game

    def update(self):
        self.timer += 1
        for star in self.stars:
            star.update()
        # Sparkles could be regenerated or updated if they were more dynamic
        # For now, they are static after generation.

    def draw(self):
        self.screen.fill(self.COLOR_BLACK)

        for star in self.stars:
            star.draw(self.screen)

        self.draw_orbit(self.screen, (self.center_x, self.center_y), self.outer_ring_radius, 3, self.COLOR_GOLD)
        self.draw_orbit(self.screen, (self.center_x, self.center_y), self.inner_ring_radius, 2, self.COLOR_DARKER) # Thinner inner ring

        self.draw_sparkles(self.screen, self.sparkles)
        
        self.draw_blocky_text(self.screen, self.main_text, self.main_text_x, self.main_text_y, scale=self.main_scale, outline_width=4) # Thicker outline

        # Blinking sub-text ("PRESS SPACE")
        blink_interval = FPS // 2 # Blink twice per second (on for 0.5s, off for 0.5s)
        if (self.timer // blink_interval) % 2 == 0:
            self.draw_blocky_text(self.screen, self.sub_text, self.sub_text_x, self.sub_text_y, scale=self.sub_scale, fill_color=self.COLOR_STARS, outline_width=2)

        pygame.display.flip()

    @staticmethod
    def draw_orbit(surface, center, radius, thickness, color, segments=120): # Fewer segments for orbit
        cx, cy = center
        points = []
        for i in range(segments):
            angle = (i / float(segments)) * 2 * math.pi
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))
        pygame.draw.lines(surface, color, True, points, thickness)


    @staticmethod
    def generate_sparkles(center_x, center_y, radius, count):
        sparkles = []
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            # Distribute sparkles more towards the periphery of the radius
            dist = radius * (0.6 + random.uniform(0, 0.4)) # Concentrate between 60% and 100% of radius
            sx = center_x + dist * math.cos(angle)
            sy = center_y + dist * math.sin(angle)
            size = random.randint(1, 2) # Smaller sparkles
            sparkles.append({'pos': (sx, sy), 'size': size, 'brightness': random.uniform(0.5, 1.0)})
        return sparkles

    @staticmethod
    def draw_sparkles(surface, sparkles):
        for sparkle in sparkles:
            pos = sparkle['pos']
            size = sparkle['size']
            # Modulate base COLOR_STARS by sparkle's brightness for variation
            color = (
                min(255, int(FinalFlashingPressStart.COLOR_STARS[0] * sparkle['brightness'])),
                min(255, int(FinalFlashingPressStart.COLOR_STARS[1] * sparkle['brightness'])),
                min(255, int(FinalFlashingPressStart.COLOR_STARS[2] * sparkle['brightness']))
            )
            # Draw as small crosses or circles
            pygame.draw.line(surface, color, (pos[0] - size, pos[1]), (pos[0] + size, pos[1]), 1)
            pygame.draw.line(surface, color, (pos[0], pos[1] - size), (pos[0], pos[1] + size), 1)


    @classmethod
    def draw_filled_letter(cls, surface, polygons, ox, oy, scale, fill_color, outline_color, outline_width):
        for poly_pts in polygons:
            scaled_points = [(ox + px * scale, oy + py * scale) for px, py in poly_pts]
            pygame.draw.polygon(surface, fill_color, scaled_points)
            pygame.draw.polygon(surface, outline_color, scaled_points, width=outline_width)

    @classmethod
    def draw_blocky_text(cls, surface, text, x, y, scale, fill_color=None, outline_color=None, outline_width=3):
        current_x = x
        default_fill = cls.COLOR_GOLD
        default_outline = cls.COLOR_DARKER

        for char_code in text:
            char_upper = char_code.upper()
            letter_polygons = cls.LETTERS.get(char_upper)

            if letter_polygons:
                # Use provided colors or fallback to defaults
                current_fill = fill_color if fill_color else default_fill
                current_outline = outline_color if outline_color else default_outline
                
                cls.draw_filled_letter(surface, letter_polygons, current_x, y, scale, current_fill, current_outline, outline_width)
            
            # Advance x position for next character
            # Spacing can be adjusted based on character width if needed
            spacing_multiplier = 22 if char_upper in ('W', 'R') else 18 # Wider chars get more space
            if char_upper == ' ': spacing_multiplier += 8 # Extra for space
            current_x += spacing_multiplier * scale
