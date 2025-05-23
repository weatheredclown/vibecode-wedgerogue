import pygame
from pygame.math import Vector2

# Screen Shake Globals
SCREEN_SHAKE_MAG = 0
SCREEN_SHAKE_DECAY = 0.9  # how quickly the shake magnitude decays

def trigger_shake(amount=10):
    """Triggers a screen shake effect."""
    global SCREEN_SHAKE_MAG
    # Only set if the new shake is larger than the current one,
    # or if you want shakes to be additive/override, adjust logic here.
    if amount > SCREEN_SHAKE_MAG:
        SCREEN_SHAKE_MAG = amount

def draw_text(surface, text, x, y, color=(255,255,255), font_size=24, center=False):
    """Renders and draws text on a surface."""
    # It's good practice to initialize font module if not done, but typically main game does pygame.init()
    # if not pygame.font.get_init():
    #     pygame.font.init()
    font = pygame.font.SysFont(None, font_size)
    lines = text.split('\n')
    offset_y = 0
    for line in lines:
        rendered = font.render(line, True, color)
        if center:
            rect = rendered.get_rect(center=(x, y + offset_y))
        else:
            rect = rendered.get_rect(topleft=(x, y + offset_y))
        surface.blit(rendered, rect)
        offset_y += font_size + 2 # Add a small gap between lines

def wrap_position(pos, width, height):
    """Wraps a position vector around the screen dimensions."""
    x, y = pos.x, pos.y # Expect pos to be a Vector2 or have .x, .y attributes
    if x < 0: 
        x = width
    elif x > width:
        x = 0
    if y < 0:
        y = height
    elif y > height:
        y = 0
    return Vector2(x, y)
