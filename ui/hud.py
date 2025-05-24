import pygame
import math

class HUDSystem:
    # Style constants defined as class attributes
    SCIFI_FONT_PATH: str = None    # Path to a .ttf font file, or None for system default
    FONT_SIZE: int = 18            # Base font size for most HUD elements
    SMALL_FONT_OFFSET: int = 6     # Small font is (FONT_SIZE - SMALL_FONT_OFFSET)

    GLOW_PHASE_SPEED: float = 0.002 # Speed of the glow animation cycle
    GLOW_ALPHA_VALUES = [50, 35, 20]  # Alpha values for layered glow effect (outer to inner)

    # Color definitions
    COLOR_TEXT = (255, 255, 255)    # Default text color
    COLOR_HP = (255, 30, 30)        # Health bar color
    COLOR_SHIELD = (30, 200, 255)   # Shield bar color
    COLOR_BOMBS = (255, 220, 0)     # Bombs text color
    COLOR_COOL = (255, 150, 0)      # Cooldown bar color (e.g., for special weapon)
    COLOR_SCORE = (150, 255, 150)   # Score text color
    COLOR_BAR_BG = (40, 40, 40)     # Background color for bars
    COLOR_OUTLINE = (80, 80, 80)    # Outline color for bars and other elements
    COLOR_SHADOW = (0, 0, 0, 100)   # Shadow color for text (includes alpha)

    # Sizing and layout parameters
    BAR_WIDTH = 200                 # Default width for status bars
    BAR_HEIGHT = 14                 # Default height for status bars
    PADDING = 28                    # Padding from screen edges and between HUD elements

    def __init__(self):
        """Initializes fonts and internal states for the HUD."""
        # Initialize fonts (attempt custom, fallback to system default)
        try:
            if self.SCIFI_FONT_PATH and pygame.font.get_init(): # Ensure font module is ready
                self.font = pygame.font.Font(self.SCIFI_FONT_PATH, self.FONT_SIZE)
                self.small_font = pygame.font.Font(self.SCIFI_FONT_PATH, self.FONT_SIZE - self.SMALL_FONT_OFFSET)
            else:
                raise pygame.error("Custom font path not provided or font module not initialized.")
        except pygame.error: # Fallback if custom font fails or not specified
            # print(f"HUD Warning: Failed to load custom font. Using system default 'bahnschrift'. Error: {e}")
            self.font = pygame.font.SysFont("bahnschrift", self.FONT_SIZE, bold=True)
            self.small_font = pygame.font.SysFont("bahnschrift", self.FONT_SIZE - self.SMALL_FONT_OFFSET, bold=True)
        
        self.glow_phase = 0.0  # Used to animate the glow effect on text

    def update(self, dt: float):
        """Updates HUD animations, like the glow effect. dt is delta time."""
        self.glow_phase += dt * self.GLOW_PHASE_SPEED
        if self.glow_phase > math.pi * 2: # Keep phase within 0 to 2*pi
            self.glow_phase -= math.pi * 2

    def draw(self, surface, player, screen_w, screen_h):
        """Draws all HUD elements onto the provided surface."""
        # Top-left: HP and Shield bars
        tl_x, tl_y = self.PADDING, self.PADDING
        self.draw_bar_with_label(surface, "HP", player.hp, player.max_hp, (tl_x, tl_y), self.COLOR_HP)
        tl_y += self.BAR_HEIGHT + self.PADDING # Move down for next element
        
        if getattr(player, "shield_max", 0) > 0: # Only draw shield bar if player has one
            self.draw_bar_with_label(surface, "SHIELD", player.shield_hp, player.shield_max, (tl_x, tl_y), self.COLOR_SHIELD)
            # tl_y += self.BAR_HEIGHT + self.PADDING # Update tl_y if more elements were to follow here

        # Top-right: Score
        score_text = f"SCORE {player.score}"
        self.draw_glow_text(surface, score_text, self.font, self.COLOR_SCORE, screen_w - self.PADDING, self.PADDING, align="top-right")

        # Bottom-left: Bombs (if applicable)
        # Safely access upgrades and bombs count
        player_upgrades = getattr(player, "upgrades", {})
        if "bombs" in player_upgrades and player_upgrades.get("bombs", 0) > 0 : # Only show if bombs upgrade exists and count > 0
            bombs_count = player_upgrades.get("bombs",0)
            bombs_text = f"BOMBS {bombs_count}"
            self.draw_glow_text(surface, bombs_text, self.font, self.COLOR_BOMBS, self.PADDING, screen_h - self.PADDING, align="bottom-left")

        # Bottom-right: Special weapon cooldown bar (if applicable)
        if "special_weapon_cooldown" in player_upgrades:
            cooldown_val = player_upgrades.get("special_weapon_cooldown", 0)
            max_cooldown = player_upgrades.get("special_weapon_max_cooldown", 100) # Assume 100 if not specified
            bar_x = screen_w - (self.BAR_WIDTH + self.PADDING)
            bar_y = screen_h - (self.BAR_HEIGHT + self.PADDING)
            self.draw_bar(surface, cooldown_val, max_cooldown, (bar_x, bar_y), self.BAR_WIDTH, self.BAR_HEIGHT, self.COLOR_COOL, label="SPECIAL", label_align="right")

    def draw_bar_with_label(self, surface, label, current, maximum, pos, color):
        """Helper to draw a status bar with a label positioned above it."""
        self.draw_bar(surface, current, maximum, pos, self.BAR_WIDTH, self.BAR_HEIGHT, color, label=label, label_align="left")

    def draw_bar(self, surface, current, maximum, pos, width, height, fill_color, label=None, label_align="left"):
        """Draws a single status bar with optional label and gradient fill."""
        x, y = pos
        # Draw bar outline and background
        pygame.draw.rect(surface, self.COLOR_OUTLINE, pygame.Rect(x - 1, y - 1, width + 2, height + 2), 1)
        pygame.draw.rect(surface, self.COLOR_BAR_BG, pygame.Rect(x, y, width, height))

        # Calculate fill width based on current/maximum values
        frac = max(0.0, min(1.0, float(current) / float(maximum) if maximum > 0 else 0.0))
        fill_w = int(width * frac)

        # Draw gradient fill for the bar
        if fill_w > 0:
            grad_surf = pygame.Surface((fill_w, height)) # Only create gradient for the filled part
            for row in range(height):
                ratio = row / float(height - 1) if height > 1 else 0.0
                r = int(fill_color[0] * (1 - ratio) + self.COLOR_BAR_BG[0] * ratio) # Fade to background color
                g = int(fill_color[1] * (1 - ratio) + self.COLOR_BAR_BG[1] * ratio)
                b = int(fill_color[2] * (1 - ratio) + self.COLOR_BAR_BG[2] * ratio)
                pygame.draw.line(grad_surf, (r, g, b), (0, row), (fill_w, row))
            surface.blit(grad_surf, (x, y))

        # Draw numeric text (current/max) centered on the bar
        val_text = f"{int(current)}/{int(maximum)}"
        text_surf = self.small_font.render(val_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(x + width // 2, y + height // 2))
        surface.blit(text_surf, text_rect) # No shadow for numbers for clarity

        # Draw label text (if provided)
        if label:
            label_surf = self.small_font.render(label.upper(), True, self.COLOR_TEXT)
            label_x = x if label_align == "left" else (x + width - label_surf.get_width())
            label_y = y - label_surf.get_height() - 2 # Position label above the bar
            # surface.blit(label_surf, (label_x + 1, label_y + 1)) # Shadow for label
            surface.blit(label_surf, (label_x, label_y))


    def draw_glow_text(self, surface, text, font, color, x, y, align="top-left"):
        """Renders text with a glowing effect."""
        text_upper = text.upper()
        base_surf = font.render(text_upper, True, color)
        
        # Determine initial rect based on alignment before creating glow_surf
        base_rect = base_surf.get_rect()
        if "right" in align: base_rect.right = x
        else: base_rect.left = x
        if "bottom" in align: base_rect.bottom = y
        else: base_rect.top = y

        # Create a larger temporary surface to accommodate the glow
        glow_radius = max(self.GLOW_ALPHA_VALUES) / 10 # Approximate glow extent
        temp_surf_size = (base_rect.width + int(glow_radius * 4) , base_rect.height + int(glow_radius * 4))
        glow_surf = pygame.Surface(temp_surf_size, pygame.SRCALPHA)
        
        # Center of the temporary surface for blitting glow layers
        center_x_temp = temp_surf_size[0] // 2
        center_y_temp = temp_surf_size[1] // 2

        current_glow_radius = 2 + 2 * math.sin(self.glow_phase)

        for i, alpha_val in enumerate(self.GLOW_ALPHA_VALUES):
            # Render text in a slightly lighter/whiter version of the color for the glow
            glow_color_val = min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50)
            temp_text_surf = font.render(text_upper, True, glow_color_val)
            temp_text_surf = temp_text_surf.convert_alpha()
            temp_text_surf.set_alpha(alpha_val)
            
            # Blit glow layers around the center of temp_surf
            # Offset slightly for a more diffuse glow, can be adjusted
            offset = current_glow_radius * (i + 1) * 0.5 
            glow_offsets = [
                (dx*offset, dy*offset) for dx in [-1,0,1] for dy in [-1,0,1] if not (dx==0 and dy==0)
            ]
            # Add a few more specific offsets for a smoother glow
            glow_offsets.extend([(offset,0),(-offset,0),(0,offset),(0,-offset)])


            for dx, dy in glow_offsets:
                 glow_surf.blit(temp_text_surf, (center_x_temp - temp_text_surf.get_width()//2 + dx, 
                                                 center_y_temp - temp_text_surf.get_height()//2 + dy))

        # Blit the main text on top
        glow_surf.blit(base_surf, (center_x_temp - base_rect.width//2, center_y_temp - base_rect.height//2))
        
        # Final blit to the main surface, aligning glow_surf correctly
        final_rect = glow_surf.get_rect()
        if "right" in align: final_rect.right = x
        else: final_rect.left = x
        if "bottom" in align: final_rect.bottom = y
        else: final_rect.top = y
        surface.blit(glow_surf, final_rect)
