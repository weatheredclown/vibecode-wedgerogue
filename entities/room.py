import random
import pygame # Not directly used for drawing by Room, but good for context if expanded

from .enemy import Enemy
from .loot import FancyLootToken
from .particles import ParticleSystem
# from ui.countdown import FancyCountdown # This will be the proper import later

from sfx import sfx_system
from config import WIDTH, HEIGHT, COLLECTION_TIME, FPS, SFX_HIT_PLAYER, SFX_HIT_ENEMY, SFX_COUNTDOWN 
# Note: FPS is used by FancyCountdown placeholder, SFX_COUNTDOWN also.
# SFX_HIT_ENEMY is used in Room.update

# Placeholder for FancyCountdown if ui/countdown.py is not yet created
try:
    from ui.countdown import FancyCountdown
except ImportError:
    class FancyCountdown:
        def __init__(self, fps=60, base_font_size=100, color=(255,255,255), font_name=None):
            self.fps = fps; self.base_font_size = base_font_size; self.color = color; self.font_name = font_name
            self.active = False; self.frames_left = 0; self.total_seconds = 0; self.seconds_left = 0; self.last_seconds_left = -1
            print("Warning: Using placeholder FancyCountdown class in entities/room.py")
        def start(self, total_seconds=10):
            self.total_seconds = total_seconds; self.frames_left = total_seconds * self.fps; self.active = True; self.seconds_left = total_seconds; self.last_seconds_left = -1
        def stop(self): self.active = False; self.frames_left = 0
        def update(self):
            if not self.active: return
            self.frames_left -= 1
            if self.frames_left < 0: self.frames_left = 0; self.seconds_left = 0; self.active = False
            else: self.seconds_left = (self.frames_left + (self.fps - 1)) // self.fps
            if self.seconds_left != self.last_seconds_left:
                if sfx_system and SFX_COUNTDOWN is not None: sfx_system.play_sound(SFX_COUNTDOWN)
                self.last_seconds_left = self.seconds_left
        def draw(self, surface):
            if not self.active or self.seconds_left < 1 or self.seconds_left > self.total_seconds: return
            # Simplified drawing for placeholder
            font_obj = pygame.font.SysFont(self.font_name, 50)
            text_surf = font_obj.render(str(self.seconds_left), True, self.color)
            surface.blit(text_surf, text_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        def is_active(self): return self.active


class Room:
    def __init__(self, idx, is_boss=False):
        self.idx = idx
        self.is_boss = is_boss
        self.enemies = []
        if not is_boss:
            for _ in range(random.randint(1,3)): 
                self.enemies.append(Enemy((random.randint(100,WIDTH-100),random.randint(100,HEIGHT-100))))
        else: 
            self.enemies.append(Enemy((WIDTH/4,HEIGHT/4),is_boss=True))
        
        self.loot = []
        self.particles = ParticleSystem()
        # FPS is needed from config for FancyCountdown
        self.countdown = FancyCountdown(fps=FPS, base_font_size=120, color=(255,255,255))

    def start_collection(self):
        # Convert COLLECTION_TIME (milliseconds) to seconds for countdown, if it's large
        collection_seconds = COLLECTION_TIME // 1000 if COLLECTION_TIME >= 1000 else COLLECTION_TIME
        self.countdown.start(collection_seconds)

    def update(self, player, global_frame):
        for e in self.enemies: 
            e.update(global_frame, self.enemies) # Pass self.enemies for potential enemy-enemy interaction (though not used by current Enemy.update)
        
        for e in self.enemies[:]: # Iterate on a copy for safe removal
            for pb in player.bullets: # Assuming player.bullets is a list of Bullet objects
                if e.hit_by(pb):
                    e.health -= 1
                    if sfx_system: sfx_system.play_sound(SFX_HIT_ENEMY)
                    if pb.exploding:
                        for oe in self.enemies: 
                            if (oe.pos - pb.pos).length_squared() < (40*40): # Check radius squared
                                oe.health -= 1 
                    pb.life = 0 # Mark player bullet for removal
                    if e.health <= 0:
                        if not e.dropped_loot:
                            e.dropped_loot = True
                            self.loot.append(FancyLootToken(e.pos, e.color, e.points, particle_system=self.particles))
                            player.score += e.points
                        if e in self.enemies: self.enemies.remove(e)
                        break # Move to next enemy as current one is dead
            if e.health <= 0 and e in self.enemies: # Double check if removed by AOE
                 if not e.dropped_loot: # Ensure loot drops even if killed by AOE not directly targeting it
                    e.dropped_loot = True
                    self.loot.append(FancyLootToken(e.pos, e.color, e.points, particle_system=self.particles))
                    player.score += e.points
                 if e in self.enemies: self.enemies.remove(e)


        self.check_enemy_bullets_hit_player(player)
        
        for l_token in self.loot[:]: # Iterate on a copy
            l_token.update()
            if l_token.check_collision(player):
                l_token.on_collected()
                if sfx_system: sfx_system.play_sound(27) # Placeholder for SFX_LOOT_COLLECTED
                player.score += l_token.value * 5 
                self.loot.remove(l_token)
                
        self.particles.update()
        self.countdown.update()

    def check_enemy_bullets_hit_player(self, player):
        for e in self.enemies:
            for b in e.bullets[:]: # Iterate on a copy
                # Assuming player has .pos and a conceptual radius of 20
                if (b.pos - player.pos).length_squared() < (b.radius + 20)**2: 
                    if player.shield_hp > 0:
                        player.shield_hp -= 1
                    else:
                        player.hp -= 1
                    if sfx_system: sfx_system.play_sound(SFX_HIT_PLAYER)
                    self.particles.spawn_hit_particles(b.pos, color=(255, 200, 50), count=8)
                    e.bullets.remove(b) # Remove bullet that hit

    def draw(self, surface):
        for e in self.enemies: e.draw(surface)
        for l_token in self.loot: l_token.draw(surface)
        self.particles.draw(surface)
        self.countdown.draw(surface)

    def is_enemy_cleared(self): 
        return len(self.enemies) == 0

    def is_collection_done(self): 
        return not self.countdown.active # Assuming FancyCountdown sets active to False when done
        # Or, if FancyCountdown has an is_done() method: return self.countdown.is_done()
