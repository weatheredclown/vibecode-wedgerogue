import pygame
from utils import draw_text
from sfx import sfx_system # Assuming sfx_system is globally accessible or passed if needed
from config import WIDTH, HEIGHT, SFX_BUY_SUCCESS, SFX_BUY_FAIL

POWERUPS = [
    {"name": "Homing Bullets",         "cost": 50,  "desc": "Bullets steer slightly towards enemies."},
    {"name": "Better Homing Bullets",  "cost": 100, "desc": "Bullets steer more aggressively."},
    {"name": "Exploding Bullets",      "cost": 80,  "desc": "Bullets explode on impact."},
    {"name": "Bigger Bullets",         "cost": 60,  "desc": "Increase bullet size and maybe damage."},
    {"name": "Ship Shield",            "cost": 120, "desc": "Absorb some hits before taking damage."},
    {"name": "Ship Teleporter",        "cost": 150, "desc": "Press [T] to teleport short distance."},
    {"name": "Short-Range Autofire",   "cost": 90,  "desc": "Automatically fires at nearby enemies."},
    {"name": "Mines",                  "cost": 70,  "desc": "Press [M] to drop stationary mines."},
    {"name": "Homing Mines",           "cost": 130, "desc": "Mines [M] slowly chase enemies."},
]

class StoreSystem:
    def __init__(self):
        self.active = False
        self.message = ""
        self.timer = 0
        self.prev_keys = None  # To store previous key states for detecting new presses
        self.items_purchased = set() # Tracks indices of purchased items

    def open_store(self):
        self.active = True
        self.message = "Store: Press [1-9] to buy power-ups, ESC/ENTER to exit"
        self.timer = 0

    def close_store(self):
        self.active = False
        self.message = ""

    def update(self, player, keys): # Player object is needed to access score and apply upgrades
        if self.prev_keys is None: # Initialize prev_keys on first update
            self.prev_keys = {
                key_code: keys[key_code] for key_code in 
                [pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_1, pygame.K_2, pygame.K_3, 
                 pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]
                if key_code < len(keys) # Ensure key_code is a valid index for keys
            }
            return

        if not self.active:
            return

        self.timer += 1

        # Close store on ESC or ENTER
        if keys[pygame.K_ESCAPE] or keys[pygame.K_RETURN]:
            self.close_store()
            return

        # Check for item purchase attempts (keys 1-9)
        for i in range(1, 10):
            key_code = getattr(pygame, f'K_{i}')
            if key_code < len(keys): # Check if key_code is valid for the keys sequence
                current_key_state = keys[key_code]
                prev_key_state = self.prev_keys.get(key_code, False) # Default to False if not found

                if current_key_state and not prev_key_state: # New press
                    self.buy_item(player, i - 1) # index is 0-8 for POWERUPS list

        # Update prev_keys for the next frame
        for key_code in self.prev_keys.keys():
             if key_code < len(keys):
                self.prev_keys[key_code] = keys[key_code]


    def buy_item(self, player, index):
        if not (0 <= index < len(POWERUPS)):
            self.message = "Invalid item selection."
            return
        
        if index in self.items_purchased:
            self.message = "You've already bought this item!"
            if sfx_system: sfx_system.play_sound(SFX_BUY_FAIL) # Play fail sound
            return

        item = POWERUPS[index]
        if player.score >= item["cost"]:
            if sfx_system: sfx_system.play_sound(SFX_BUY_SUCCESS)
            player.score -= item["cost"]
            self.apply_powerup(player, item["name"]) # Pass item name for clarity
            self.items_purchased.add(index)
            self.message = f"Bought {item['name']} for {item['cost']} pts!"
        else:
            if sfx_system: sfx_system.play_sound(SFX_BUY_FAIL)
            self.message = f"Not enough points for {item['name']} (cost {item['cost']})."

    def apply_powerup(self, player, item_name):
        # This method assumes 'player' has an 'upgrades' dictionary
        # and attributes like 'shield_max', 'shield_hp'.
        if item_name == "Homing Bullets":
            player.upgrades["homing_bullet_level"] = max(player.upgrades.get("homing_bullet_level", 0), 1)
        elif item_name == "Better Homing Bullets":
            player.upgrades["homing_bullet_level"] = 2
        elif item_name == "Exploding Bullets":
            player.upgrades["exploding_bullets"] = True
        elif item_name == "Bigger Bullets":
            player.upgrades["bigger_bullets"] = True
        elif item_name == "Ship Shield":
            player.upgrades["shield"] = True
            player.shield_max = getattr(player, 'shield_max_base', 30) # Assuming a base max shield
            player.shield_hp = player.shield_max
        elif item_name == "Ship Teleporter":
            player.upgrades["teleporter"] = True
        elif item_name == "Short-Range Autofire":
            player.upgrades["short_range_autofire"] = True
        elif item_name == "Mines":
            player.upgrades["mines"] = True
        elif item_name == "Homing Mines":
            player.upgrades["mines"] = True # Homing mines also grants basic mines
            player.upgrades["homing_mines"] = True
        # Add effects for other power-ups as needed

    def draw(self, surface):
        if not self.active:
            return

        # Draw store message (e.g., "Store open", item purchased, not enough points)
        draw_text(surface, self.message, WIDTH // 2, HEIGHT // 2 - 200, center=True, font_size=24)
        
        # List available power-ups
        y_offset = HEIGHT // 2 - 140
        for i, item in enumerate(POWERUPS):
            item_text = f"{i+1}) {item['name']} ({item['cost']} pts) - {item['desc']}"
            color = (120, 120, 120) if i in self.items_purchased else (255, 255, 0) # Dim purchased items
            draw_text(surface, item_text, WIDTH // 2, y_offset, color=color, font_size=22, center=True)
            y_offset += 30
