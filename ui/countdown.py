import pygame
import math # Not strictly used in the provided snippet but often useful for UI animations
from sfx import sfx_system 
from config import SFX_COUNTDOWN, WIDTH, HEIGHT # WIDTH, HEIGHT for centering

class FancyCountdown:
    """
    A fancy countdown timer that displays shrinking, fading numbers.
    """
    def __init__(self, fps=60, base_font_size=100, color=(255,255,255), font_name=None):
        self.fps = fps
        self.base_font_size = base_font_size
        self.color = color
        self.font_name = font_name # Allows specifying a font, None for pygame default

        self.active = False
        self.frames_left = 0
        self.total_seconds_for_current_countdown = 0 # Store the initial total seconds
        self.seconds_left_in_current_segment = 0 # Number currently being displayed
        self.last_displayed_second = -1 # Tracks the number last shown to trigger sound

    def start(self, total_seconds=10):
        """Begins or restarts the countdown from total_seconds down to 1."""
        self.total_seconds_for_current_countdown = total_seconds
        self.frames_left = total_seconds * self.fps
        self.active = True
        # Initial calculation for seconds_left_in_current_segment
        self.seconds_left_in_current_segment = (self.frames_left + (self.fps - 1)) // self.fps
        self.last_displayed_second = self.seconds_left_in_current_segment + 1 # Ensure sound plays for the first number

    def stop(self):
        """Manually stops the countdown."""
        self.active = False
        self.frames_left = 0

    def update(self):
        """Updates the countdown timer by one frame."""
        if not self.active:
            return

        self.frames_left -= 1
        if self.frames_left < 0:
            self.frames_left = 0
            self.seconds_left_in_current_segment = 0 # Ensure it shows 0 if timer ends exactly
            self.active = False
        else:
            # Calculate the current number to display (e.g., 3, 2, 1)
            self.seconds_left_in_current_segment = (self.frames_left + (self.fps - 1)) // self.fps
        
        # Play sound when the displayed number changes
        if self.active and self.seconds_left_in_current_segment != self.last_displayed_second and self.seconds_left_in_current_segment > 0:
            if sfx_system and SFX_COUNTDOWN is not None:
                sfx_system.play_sound(SFX_COUNTDOWN)
            self.last_displayed_second = self.seconds_left_in_current_segment
        elif not self.active and self.last_displayed_second != 0: # Ensure sound plays for "1" if it just finished
             if sfx_system and SFX_COUNTDOWN is not None and self.last_displayed_second == 1 : # Check if the sound for 1 needs to play
                 pass # Sound for 1 already played when it became the current second
             self.last_displayed_second = 0


    def draw(self, surface):
        """ 
        If active and a number is to be displayed (1 to total_seconds),
        draws the current second with a fade/scale effect.
        """
        if not self.active or self.seconds_left_in_current_segment < 1 or \
           self.seconds_left_in_current_segment > self.total_seconds_for_current_countdown:
            return

        # frames_within_this_second: how many frames we are *into* displaying the current number.
        # If seconds_left_in_current_segment is 3, this is 0 when 3 just appeared, up to self.fps-1.
        frames_elapsed_for_this_number = (self.total_seconds_for_current_countdown - self.seconds_left_in_current_segment) * self.fps
        current_second_start_frame = self.total_seconds_for_current_countdown * self.fps - self.seconds_left_in_current_segment * self.fps
        
        # Correct calculation for how far we are into displaying the current number
        # (from 0 to self.fps -1 for each number segment)
        frames_into_current_number_display = self.frames_left - (self.seconds_left_in_current_segment -1) * self.fps if self.seconds_left_in_current_segment > 0 else 0
        frames_into_current_number_display = self.fps - (self.frames_left % self.fps if self.frames_left % self.fps != 0 else self.fps)


        # fraction_of_second_completed: 0 means number just appeared, 1 means it's about to change.
        # This makes the animation go from big/transparent to small/opaque.
        fraction_of_second_completed = frames_into_current_number_display / float(self.fps)
        
        # Animation: scale from large to normal, alpha from transparent to opaque
        current_scale = 2.0 - (1.0 * fraction_of_second_completed) # Scale from 2.0 down to 1.0
        current_alpha = int(255 * fraction_of_second_completed)   # Alpha from 0 up to 255

        current_scale = max(1.0, current_scale) # Clamp scale
        current_alpha = max(0, min(255, current_alpha)) # Clamp alpha

        font_size = int(self.base_font_size * current_scale)
        if font_size < 10: font_size = 10 # Minimum font size

        try:
            font_obj = pygame.font.SysFont(self.font_name, font_size)
            text_str = str(self.seconds_left_in_current_segment)
            text_surf = font_obj.render(text_str, True, self.color)
            text_surf = text_surf.convert_alpha() # Essential for per-pixel alpha
            text_surf.set_alpha(current_alpha)

            # Center the text on the screen (using WIDTH, HEIGHT from config)
            center_x = WIDTH // 2
            center_y = HEIGHT // 2
            rect = text_surf.get_rect(center=(center_x, center_y))
            surface.blit(text_surf, rect)
        except Exception as e:
            print(f"Error drawing countdown text: {e}")

    def is_active(self): # Convenience method
        return self.active
