import asyncio
import os
import math # Retained for general math functions if any are used by classes still in main.py (e.g. FinalFlashingPressStart was moved)
import random # Retained for similar reasons.
import sys
# import numpy as np # Removed as numpy usage is now in sfx.py
import pygame
from pygame.math import Vector2

from config import * # Import all from config.py
from sfx import SfxSystem
from utils import draw_text, wrap_position, SCREEN_SHAKE_MAG, SCREEN_SHAKE_DECAY, trigger_shake
from entities.enemy import Enemy 
from entities.player import PlayerShip
from entities.room import Room 
from entities.bullet import Bullet # Explicitly imported though used by other entity classes
from entities.particles import ParticleSystem # Explicitly imported though used by other entity classes
from entities.loot import FancyLootToken, LootToken # Explicitly imported
from entities.trail import TrailSegment # Explicitly imported

from ui.hud import HUDSystem
from ui.dialog import DialogSystem # g_pre_conversations, g_post_dialogs are in ui.dialog
from ui.store import StoreSystem, POWERUPS # POWERUPS is now imported from ui.store
from ui.star_map import StarMapSystem
from ui.splash_screen import FinalFlashingPressStart
from ui.countdown import FancyCountdown # Imported as it's used by Room
from ui.options_menu import OptionsMenu
from ui.instructions import draw_instructions
from starfield import Starfield # Starfield and its Star class are in starfield.py

# Global sfx_system variable, initialized in main()
sfx_system = None 
# Global screen shake variables are imported from utils.py

async def main():
    global sfx_system 
    # global SCREEN_SHAKE_MAG # Not needed as a global declaration here; it's imported from utils

    pygame.init()
    if not pygame.font.get_init(): # Ensure font module is initialized
        pygame.font.init()
    
    sfx_system = SfxSystem() # Initialize SfxSystem after pygame.init()

    # Run Splash Screen
    splash = FinalFlashingPressStart() # This class is now imported
    splash.run() 

    # Initialize music if enabled
    if MUSIC_ENABLED: 
        try:
            pygame.mixer.music.load("assets/cq1.wav") 
            pygame.mixer.music.play(-1) 
        except pygame.error as e:
            print(f"Main Game Warning: Could not load or play music: {e}")      

    screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
    pygame.display.set_caption("WedgeRogue Refactored")
    clock = pygame.time.Clock()
    
    # Initialize game objects
    player = PlayerShip() 
    starfield = Starfield(100, player) 
    
    rooms=[]
    for i in range(NUM_ROOMS): 
        rooms.append(Room(i, is_boss=(i==NUM_ROOMS-1))) 
    
    dialog = DialogSystem()
    store  = StoreSystem() 
    star_map = StarMapSystem(NUM_ROOMS) 
    hud = HUDSystem()
    options_menu = OptionsMenu()
    
    current_room_idx = 0
    game_state = STATE_PRE_DIALOG # Uses string state from config.py
    dialog.start_random_pre_dialog()
    
    global_frame = 0
    running = True
    
    screen_offset_x = 0 
    screen_offset_y = 0

    while running:
        dt_ms = clock.tick(FPS) 
        actual_dt_seconds = dt_ms / 1000.0
        global_frame += 1
        
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if options_menu.active: 
                        options_menu.close()
                    elif game_state not in (STATE_STORE, STATE_STAR_MAP): 
                        running = False
                elif event.key == pygame.K_SPACE:
                    if game_state in (STATE_COMBAT, STATE_COLLECTION): 
                        if sfx_system: sfx_system.play_sound(SFX_SHOOT) # SFX_SHOOT is int from config
                        player.shoot() 
                elif event.key==pygame.K_o:
                    if not options_menu.active: options_menu.open()
                    else: options_menu.close()
        
        # Update game logic
        if game_state == STATE_END: 
            pass 
        else:
            if options_menu.active:
                options_menu.update(keys) # Pass current keys state
            else: 
                star_map.update(actual_dt_seconds) 
                store.update(player, keys) # Pass current keys state
                
                if dialog.is_pre_active() or dialog.is_post_active():
                    dialog.update(keys, player) # Pass current keys state

                if game_state == STATE_PRE_DIALOG:
                    if dialog.is_pre_done(): game_state = STATE_COMBAT
                elif game_state == STATE_COMBAT:
                    current_room = rooms[current_room_idx]
                    starfield.update()
                    player.update(keys, current_room.enemies)
                    current_room.update(player, global_frame)
                    if player.hp <= 0: game_state = STATE_END
                    elif current_room.is_enemy_cleared():
                        if not current_room.countdown.active: current_room.start_collection()
                        game_state = STATE_COLLECTION
                elif game_state == STATE_COLLECTION:
                    current_room = rooms[current_room_idx]
                    starfield.update()
                    player.update(keys, current_room.enemies) 
                    current_room.update(player, global_frame)
                    if player.hp <= 0: game_state = STATE_END
                    elif current_room.is_collection_done():
                        dialog.start_post_dialog(); game_state = STATE_POST_DIALOG
                elif game_state == STATE_POST_DIALOG:
                    if dialog.is_post_done(): store.open_store(); game_state = STATE_STORE
                elif game_state == STATE_STORE:
                    if not store.active:
                        prev_idx = current_room_idx
                        current_room_idx = (current_room_idx + 1) % NUM_ROOMS
                        if current_room_idx == 0 and NUM_ROOMS > 0: 
                            print("All rooms cleared! Looping or ending...")
                        star_map.open_map(prev_idx, current_room_idx)
                        game_state = STATE_STAR_MAP
                        player.reset() 
                elif game_state == STATE_STAR_MAP:
                    if not star_map.active:
                        dialog.start_random_pre_dialog(); game_state = STATE_PRE_DIALOG
        
        # Screen Shake (conceptual rendering offset)
        screen_offset_x, screen_offset_y = 0, 0
        if SCREEN_SHAKE_MAG > 0: # Uses imported SCREEN_SHAKE_MAG from utils
            screen_offset_x = random.randint(-int(SCREEN_SHAKE_MAG), int(SCREEN_SHAKE_MAG))
            screen_offset_y = random.randint(-int(SCREEN_SHAKE_MAG), int(SCREEN_SHAKE_MAG))
            # Decay logic should be handled within utils.py or via a function call to it.
            # For this refactor, direct modification of an imported global is avoided here.
            # Example: utils.decay_shake() could be called here.
            
        # Rendering
        render_surface = screen # Default to drawing directly on the main screen
        
        # If shaking, create a temporary surface to draw onto, then blit with offset
        # This is a more robust way to handle screen shake if all elements should shake.
        # However, for simplicity of this refactor, direct drawing to screen with offsetted
        # elements would be an alternative if specific items shake.
        # The current code implies the whole screen might shake.
        # A common pattern is to have a "camera" object that applies offsets.
        
        # Create a temporary surface for screen shake if active
        temp_surface = None
        if screen_offset_x != 0 or screen_offset_y != 0:
            temp_surface = surface.copy() # Or a new surface of the same size
            temp_surface.fill((0,0,0)) # Clear temp surface
            render_target = temp_surface
        else:
            render_target = screen
            screen.fill((0,0,0)) # Clear main screen if not shaking

        if game_state == STATE_STAR_MAP: 
            star_map.draw(render_target)
        else:
            starfield.draw(render_target)
            if 0 <= current_room_idx < len(rooms): # Ensure current_room_idx is valid
                 cur_room = rooms[current_room_idx]
                 cur_room.draw(render_target)
            player.draw(render_target) 
            
            if dialog.is_pre_active() or dialog.is_post_active():
                dialog.draw(render_target)
            if store.active:
                 store.draw(render_target)
            
            hud.draw(render_target, player, WIDTH, HEIGHT) 
            if 0 <= current_room_idx < len(rooms):
                draw_instructions(render_target, game_state, rooms[current_room_idx]) 
        
        if options_menu.active:
            options_menu.draw(render_target)

        if game_state == STATE_END:
            draw_text(render_target,"GAME OVER!",WIDTH//2,HEIGHT//2,color=(255,50,50),font_size=48,center=True) 
            draw_text(render_target,"Press ESC to Quit",WIDTH//2,HEIGHT//2+60,color=(255,255,255),font_size=32,center=True) 
        
        if temp_surface:
            screen.blit(temp_surface, (screen_offset_x, screen_offset_y))

        pygame.display.flip()

    if os.environ.get('PYGBAG') is None: 
        pygame.quit()
        sys.exit()

asyncio.run(main())
