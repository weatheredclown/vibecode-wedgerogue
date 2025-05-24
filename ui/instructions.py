from utils import draw_text
from config import (HEIGHT, STATE_PRE_DIALOG, STATE_POST_DIALOG, STATE_STORE, 
                    STATE_COMBAT, STATE_COLLECTION)
# Note: current_room object passed to draw_instructions is expected to have:
# - .is_enemy_cleared() method
# - .countdown object with .active and .seconds_left attributes.
# This dependency is handled by the Room class (currently in main.py, later to be moved).

def draw_instructions(surface, game_state, current_room):
    """
    Displays game instructions and context-sensitive help text on the screen.
    """
    lines = []
    lines.append("Controls:")
    lines.append("  Arrow Keys = Move/Rotate")
    lines.append("  SPACE = Shoot")
    lines.append("  T = Teleport (if upgraded)")
    lines.append("  M = Drop Mines (if upgraded)")
    lines.append("  P = Toggle Control Mode (Classic/Alt)")
    lines.append("  O = Options Menu")


    if game_state == STATE_PRE_DIALOG:
        lines.append("Press ENTER to skip dialog.")
    elif game_state == STATE_POST_DIALOG:
        lines.append("Choose an option (1 or 2).")
    elif game_state == STATE_STORE:
        lines.append("Press 1-9 to buy, ESC/ENTER to exit.")
    elif game_state == STATE_COMBAT:
        if current_room and not current_room.is_enemy_cleared():
            lines.append("Objective: Defeat all enemies!")
        elif current_room: # Enemies cleared, but not yet in collection (should not happen if state logic is correct)
             lines.append("All enemies cleared!")
    elif game_state == STATE_COLLECTION:
        if current_room and current_room.countdown.active:
            seconds_left = int(current_room.countdown.seconds_left)
            lines.append(f"Collect loot! Time left: {seconds_left}s")
        else:
            lines.append("Loot collection complete.")
    
    # Positioning and rendering
    start_x = 15
    start_y = HEIGHT - (len(lines) * 20 + 10) # Dynamic y based on number of lines
    if start_y < HEIGHT * 0.6: # Ensure it doesn't go too high
        start_y = int(HEIGHT * 0.6)

    for i, line in enumerate(lines):
        # Highlight key controls or important info differently
        color = (200, 200, 220) # Default light color
        if "Controls:" in line or "Objective:" in line:
            color = (255, 255, 100) # Yellow for headers
        elif "[" in line and "]" in line:
             color = (100, 255, 100) # Green for keybinds

        draw_text(surface, line, start_x, start_y + (i * 20), 
                  color=color, font_size=18, center=False)
