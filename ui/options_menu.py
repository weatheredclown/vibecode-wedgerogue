import pygame
from config import toggle_music, toggle_sfx, WIDTH, HEIGHT
from utils import draw_text

class OptionsMenu:
    def __init__(self):
        self.active = False
        self.timer = 0 # Not currently used, but kept from original
        self.prev_keys = { # To detect single key presses for toggles
            pygame.K_m: False,
            pygame.K_s: False,
            pygame.K_ESCAPE: False
        }


    def open(self):
        self.active = True
        # Reset key states when opening to avoid instant toggles if keys were held
        for key_code in self.prev_keys:
            self.prev_keys[key_code] = True # Assume keys are up when menu opens

    def close(self):
        self.active = False

    def update(self, keys):
        if not self.active:
            return

        # Toggle music with 'M' key (on new press)
        current_m_pressed = keys[pygame.K_m]
        if current_m_pressed and not self.prev_keys[pygame.K_m]:
            toggle_music()
        self.prev_keys[pygame.K_m] = current_m_pressed

        # Toggle SFX with 'S' key (on new press)
        current_s_pressed = keys[pygame.K_s]
        if current_s_pressed and not self.prev_keys[pygame.K_s]:
            toggle_sfx()
        self.prev_keys[pygame.K_s] = current_s_pressed
        
        # Close menu with ESC key (on new press)
        current_esc_pressed = keys[pygame.K_ESCAPE]
        if current_esc_pressed and not self.prev_keys[pygame.K_ESCAPE]:
            self.close()
        self.prev_keys[pygame.K_ESCAPE] = current_esc_pressed


    def draw(self, surface):
        if not self.active:
            return
        
        # Draw a semi-transparent overlay
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180)) # Slightly darker overlay
        surface.blit(overlay, (0,0))

        # Menu text
        lines = [
            "OPTIONS",
            "",
            "[M] Toggle Music",
            "[S] Toggle SFX",
            "",
            "[ESC] Return to Game"
        ]
        
        title_font_size = 40
        option_font_size = 28
        
        # Draw title
        draw_text(surface, lines[0], WIDTH // 2, HEIGHT // 3, 
                  color=(220, 220, 255), font_size=title_font_size, center=True)
        
        # Draw options
        y_start = HEIGHT // 3 + title_font_size + 30 # Start options below title
        for i, line in enumerate(lines[2:]): # Skip title and blank line
            color = (200,200,200) if "Return" not in line else (255,200,0) # Highlight return option
            font_size = option_font_size
            if line == "": # Extra spacing for blank lines
                y_start += option_font_size // 2
                continue
            draw_text(surface, line, WIDTH // 2, y_start, 
                      color=color, font_size=font_size, center=True)
            y_start += font_size + 15 # Spacing between options
