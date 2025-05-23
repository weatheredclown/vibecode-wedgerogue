import asyncio
import os
import math
import random
import sys
import numpy as np # Uncommented numpy
import pygame
from pygame.math import Vector2
from config import * # Import all from config.py
from sfx import SfxSystem # Import SfxSystem
from utils import draw_text, wrap_position, SCREEN_SHAKE_MAG, SCREEN_SHAKE_DECAY, trigger_shake
from entities.enemy import Enemy 
from entities.player import PlayerShip # Import PlayerShip class
# Bullet will be imported by Enemy and PlayerShip from entities.bullet

# POWERUPS list remains in main.py for now, as it's used by StoreSystem also in main.py
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

################################################################################
# GAME ENTITIES
################################################################################
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
    COLOR_STARS  = (255, 200,  50)

    # Letter polygons dictionary
    LETTERS = {
        'C': [[(15,0),(3,0),(0,3),(0,37),(3,40),(15,40),(15,30),(10,30),(10,10),(15,10)]],
        'W': [[ (0, 0),(  0, 45),(5, 45),(8, 25),(12, 25),(15, 45),(20, 45),(20, 0),(16.199999999999996, 0.3),(15.099999999999998, 15.100000000000014),(10.199999999999994, 9.999999999999995),(6.500000000000001, 14.800000000000004),(4.4, 0.09999999999999731)] ],
        'E': [[(0,0),(0,40),(15,40),(15,32),(5,32),(5,22),(12,22),(12,18),(5,18),(5,8),(15,8),(15,0)]],
        'D': [[(0,0),(0,40),(11,40),(17,34),(17,6),(11,0)]],
        'G': [[(5,0),(15,0),(20,5),(20,12),(15,12),(15,9),(12,9),(12,15),(15,15),(15,18),(20,18),(20,33),(15,40),(5,40),(0,33),(0,5)]],
        'R': [[(0,45),(0,0),(13,0),(18,5),(18,15),(13,20),(4,20),(4,25),(9,25),(20,45),(15,45),(10,35),(5,35)]],
        'O': [[(4,0),(11,0),(17,6),(17,34),(11,40),(4,40),(0,34),(0,6)]],
        'U': [[(0,0),(0,35),(5,40),(12,40),(17,35),(17,0)]],
        ' ': [],
        'P': [[(0,40),(0,0),(10,0),(15,5),(15,15),(10,20),(0,20)]],
        'S': [[(15,0),(5,0),(0,5),(0,15),(5,20),(10,20),(15,25),(15,35),(10,40),(0,40)]],
        'T': [[(0,0),(15,0),(15,10),(10,10),(10,40),(5,40),(5,10),(0,10)]],
        'A': [[(0,40),(5,0),(10,0),(15,40),(10,35),(5,35)]]
    }

    class Star:
        def __init__(self, layer=0):
            self.x = random.randrange(0, WIDTH) 
            self.y = random.randrange(0, HEIGHT)
            self.layer = layer
            self.speed = 0.5 + layer*0.5
            self.base_brightness = random.randint(100, 255)
            self.brightness = self.base_brightness
            self.color = (self.brightness,)*3
            self.twinkle_timer = random.randrange(0, 100)

        def update(self):
            self.y += self.speed
            if self.y > HEIGHT: 
                self.y = 0
                self.x = random.randrange(0, WIDTH)
            twinkle_speed = random.uniform(0.01, 0.05)
            self.twinkle_timer += 1
            raw_bright = self.base_brightness + 30*math.sin(self.twinkle_timer*twinkle_speed)
            raw_bright = max(0, min(255, raw_bright))
            self.brightness = int(raw_bright)
            self.color = (self.brightness,)*3

        def draw(self, surface):
            size = 1 + self.layer
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), size)

    def __init__(self):
        # MUSIC_ENABLED, WIDTH, HEIGHT are from config now
        if MUSIC_ENABLED: 
            pygame.mixer.music.load("assets/cq1.wav")
            pygame.mixer.music.play(-1)        

        self.screen=pygame.display.set_mode((WIDTH, HEIGHT)) 
        pygame.display.set_caption("WEDGEROGUE + FLASHING PRESS SPACE with starfield & sparkles")
        self.clock=pygame.time.Clock()
        self.running=True
        self.stars=[]
        for layer in range(3):
            for _ in range(100):
                self.stars.append(self.Star(layer))
        self.outer_ring=150
        self.inner_ring=110
        self.center_x = WIDTH//2 
        self.center_y = HEIGHT//2
        self.sparkles = self.generate_sparkles(self.center_x, self.center_y, radius=140, count=20)
        self.main_text  = "WEDGEROGUE"
        self.sub_text   = "PRESS SPACE"
        self.main_scale = 2.5
        self.sub_scale  = 1.0
        self.main_text_x = self.center_x - 220
        self.main_text_y = self.center_y - 60
        self.sub_text_x  = self.center_x - 100
        self.sub_text_y  = self.center_y + 60
        self.timer=0

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) # FPS from config
            self.handle_events()
            self.update()
            self.draw()

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
                if os.environ.get('PYGBAG') is None: pygame.quit(); sys.exit()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                    if os.environ.get('PYGBAG') is None: pygame.quit(); sys.exit()
                if e.key in (pygame.K_SPACE, pygame.K_RETURN):
                    self.running = False

    def update(self):
        self.timer += 1
        for s in self.stars: s.update()

    def draw(self):
        self.screen.fill((0,0,0))
        for s in self.stars: s.draw(self.screen)
        self.draw_orbit(self.screen, (self.center_x, self.center_y), self.outer_ring, 3, self.COLOR_GOLD)
        self.draw_orbit(self.screen, (self.center_x, self.center_y), self.inner_ring, 3, self.COLOR_DARKER)
        self.draw_sparkles(self.screen, self.sparkles)
        self.draw_blocky_text(self.screen, self.main_text, self.main_text_x, self.main_text_y, scale=self.main_scale)
        if (self.timer // 40) % 2 == 0:
            self.draw_blocky_text(self.screen, self.sub_text, self.sub_text_x, self.sub_text_y, scale=self.sub_scale)
        pygame.display.flip()

    @staticmethod
    def draw_orbit(surface, center, radius, thickness=3, color=(255,200,50), segments=180):
        cx, cy = center; old_x = cx + radius; old_y = cy
        step_angle = (2*math.pi)/segments
        for i in range(1, segments+1):
            angle = i*step_angle; nx = cx + radius*math.cos(angle); ny = cy + radius*math.sin(angle)
            pygame.draw.line(surface, color, (old_x,old_y), (nx,ny), thickness)
            old_x, old_y = nx, ny

    @staticmethod
    def generate_sparkles(center_x, center_y, radius=120, count=20):
        sparkles=[]
        for _ in range(count):
            angle = random.uniform(0, 2*math.pi); dist  = random.uniform(20, radius)
            sx = center_x + dist*math.cos(angle); sy = center_y + dist*math.sin(angle)
            size = random.randint(1,3); sparkles.append((sx,sy,size))
        return sparkles

    @staticmethod
    def draw_sparkles(surface, sparkles):
        for (sx,sy,size) in sparkles:
            pygame.draw.line(surface, FinalFlashingPressStart.COLOR_STARS, (sx-size, sy), (sx+size, sy), 1)
            pygame.draw.line(surface, FinalFlashingPressStart.COLOR_STARS, (sx, sy-size), (sx, sy+size), 1)

    @classmethod
    def draw_filled_letter(cls, surface, polygons, ox, oy, scale, fill_color=None, outline_color=None, outline_width=3):
        if fill_color is None: fill_color = cls.COLOR_GOLD
        if outline_color is None: outline_color = cls.COLOR_DARKER
        for poly in polygons:
            pts=[]; 
            for(px,py) in poly: sx = ox+px*scale; sy = oy+py*scale; pts.append((sx,sy))
            pygame.draw.polygon(surface, fill_color, pts)
            pygame.draw.polygon(surface, outline_color, pts, width=outline_width)

    @classmethod
    def draw_blocky_text(cls, surface, text, x, y, scale=2.0, fill_color=None, outline_color=None, outline_width=3):
        if fill_color is None: fill_color=cls.COLOR_GOLD
        if outline_color is None: outline_color=cls.COLOR_DARKER
        normal_spacing=18; wide_spacing=22; cx=x
        for ch in text:
            if ch==' ': cx+=(normal_spacing+8)*scale; continue
            up=ch.upper(); polys=cls.LETTERS.get(up)
            if not polys: cx+=normal_spacing*scale; continue
            cls.draw_filled_letter(surface, polys, cx, y, scale, fill_color, outline_color, outline_width)
            if up in ('W','R'): cx+=wide_spacing*scale
            else: cx+=normal_spacing*scale

class Particle:
    def __init__(self, pos, vel, color, shape='circle', lifetime=30, radius=3):
        self.pos = Vector2(pos); self.vel = Vector2(vel); self.color = color
        self.shape = shape; self.lifetime = lifetime; self.max_lifetime = lifetime
        self.radius = radius; self.timer = 0

    def update(self):
        self.pos += self.vel; self.lifetime -= 1; self.timer += 1

    def draw(self, surface):
        alpha_ratio = self.lifetime / self.max_lifetime
        c = (int(self.color[0] * alpha_ratio), int(self.color[1] * alpha_ratio), int(self.color[2] * alpha_ratio))
        if self.shape == 'circle':
            pygame.draw.circle(surface, c, (int(self.pos.x), int(self.pos.y)), self.radius)
        elif self.shape == 'star':
            num_spikes = 5; points = []
            angle = self.timer * 0.3 
            for i in range(num_spikes * 2):
                r = self.radius if (i % 2 == 0) else (self.radius * 0.5)
                theta = angle + (math.pi * 2 * i) / (num_spikes * 2)
                x = self.pos.x + r * math.cos(theta); y = self.pos.y + r * math.sin(theta)
                points.append((x, y))
            pygame.draw.polygon(surface, c, points)

    def is_dead(self): return self.lifetime <= 0

class ParticleSystem:
    def __init__(self): self.particles = []
    def spawn_hit_particles(self, pos, color=(255, 200, 50), count=8):
        for _ in range(count):
            angle = random.uniform(0, 2*math.pi); speed = random.uniform(2, 5)
            vx = speed * math.cos(angle); vy = speed * math.sin(angle)
            self.particles.append(Particle(pos, (vx, vy), color, shape='circle', lifetime=20, radius=3))
    def spawn_star_burst(self, pos, color=(255, 255, 0), count=8):
        for _ in range(count):
            angle = random.uniform(0, 2*math.pi); speed = random.uniform(1.5, 4.0)
            vx = speed * math.cos(angle); vy = speed * math.sin(angle)
            self.particles.append(Particle(pos, (vx, vy), color, shape='star', lifetime=30, radius=6))
    def update(self):
        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if not p.is_dead()]
    def draw(self, surface):
        for p in self.particles: p.draw(surface)

# Bullet class definition removed from here

class HUDSystem:
    SCIFI_FONT_PATH: str = None; FONT_SIZE: int = 18; SMALL_FONT_OFFSET: int = 6
    GLOW_PHASE_SPEED: float = 0.002; GLOW_ALPHA_VALUES = [50, 35, 20]
    COLOR_TEXT = (255,255,255); COLOR_HP = (255,30,30); COLOR_SHIELD = (30,200,255)
    COLOR_BOMBS = (255,220,0); COLOR_COOL = (255,150,0); COLOR_SCORE = (150,255,150)
    COLOR_BAR_BG = (40,40,40); COLOR_OUTLINE = (80,80,80); COLOR_SHADOW = (0,0,0,100)
    BAR_WIDTH = 200; BAR_HEIGHT = 14; PADDING = 28

    def __init__(self):
        if self.SCIFI_FONT_PATH:
            self.font = pygame.font.Font(self.SCIFI_FONT_PATH, self.FONT_SIZE)
            self.small_font = pygame.font.Font(self.SCIFI_FONT_PATH, self.FONT_SIZE - self.SMALL_FONT_OFFSET)
        else:
            self.font = pygame.font.SysFont("bahnschrift", self.FONT_SIZE, bold=True)
            self.small_font = pygame.font.SysFont("bahnschrift", self.FONT_SIZE - self.SMALL_FONT_OFFSET, bold=True)
        self.glow_phase = 0.0

    def update(self, dt: float):
        self.glow_phase += dt * self.GLOW_PHASE_SPEED
        if self.glow_phase > math.pi * 2: self.glow_phase -= math.pi * 2

    def draw(self, surface, player, screen_w, screen_h):
        tl_x = self.PADDING; tl_y = self.PADDING
        self.draw_bar_with_label(surface, "HP", player.hp, player.max_hp, (tl_x, tl_y), self.COLOR_HP)
        tl_y += self.BAR_HEIGHT + self.PADDING
        if getattr(player, "shield_max", 0) > 0:
            self.draw_bar_with_label(surface, "SHIELD", player.shield_hp, player.shield_max, (tl_x, tl_y), self.COLOR_SHIELD)
        self.draw_glow_text(surface, f"SCORE {player.score}", self.font, self.COLOR_SCORE, screen_w - self.PADDING, self.PADDING, "top-right")
        if "bombs" in getattr(player, "upgrades", {}):
            self.draw_glow_text(surface, f"BOMBS {player.upgrades['bombs']}", self.font, self.COLOR_BOMBS, self.PADDING, screen_h - self.PADDING, "bottom-left")
        if "special_weapon_cooldown" in getattr(player, "upgrades", {}):
            val = player.upgrades["special_weapon_cooldown"]
            self.draw_bar(surface, val, 100, (screen_w-(self.BAR_WIDTH+self.PADDING), screen_h-(self.BAR_HEIGHT+self.PADDING)), self.BAR_WIDTH, self.BAR_HEIGHT, self.COLOR_COOL, "SPECIAL", "right")

    def draw_bar_with_label(self, surface, label, current, maximum, pos, color):
        self.draw_bar(surface, current, maximum, pos, self.BAR_WIDTH, self.BAR_HEIGHT, color, label, "left")

    def draw_bar(self, surface, current, maximum, pos, width, height, fill_color, label=None, label_align="left"):
        x,y = pos
        pygame.draw.rect(surface, self.COLOR_OUTLINE, pygame.Rect(x-1,y-1,width+2,height+2), 1)
        pygame.draw.rect(surface, self.COLOR_BAR_BG, pygame.Rect(x,y,width,height))
        frac = max(0.0, min(1.0, float(current)/float(maximum) if maximum > 0 else 0.0))
        fill_w = int(width * frac)
        if fill_w > 0:
            grad_surf = pygame.Surface((width, height))
            for row in range(height):
                rf = row/float(height-1) if height > 1 else 0; r_val,g_val,b_val = [int(c*(1-rf)+20*rf) for c in fill_color] # Renamed to avoid conflict
                pygame.draw.line(grad_surf, (r_val,g_val,b_val), (0,row), (width,row))
            surface.blit(grad_surf, (x,y), area=pygame.Rect(0,0,fill_w,height))
        val_text = f"{int(current)}/{int(maximum)}"; text_surf = self.small_font.render(val_text, True, self.COLOR_TEXT)
        cx = x+width//2-text_surf.get_width()//2; cy = y+height//2-text_surf.get_height()//2
        surface.blit(text_surf, (cx+1,cy+1)); surface.blit(text_surf, (cx,cy))
        if label:
            label_surf=self.small_font.render(label,True,self.COLOR_TEXT)
            lx = x if label_align=="left" else x+width-label_surf.get_width()
            ly = y-label_surf.get_height()-2
            surface.blit(label_surf,(lx+1,ly+1)); surface.blit(label_surf,(lx,ly))

    def draw_glow_text(self, surface, text, font, color, x, y, align="top-left"):
        base_surf = font.render(text.upper(), True, color); rect = base_surf.get_rect()
        if "right" in align: rect.right = x
        else: rect.left = x
        if "bottom" in align: rect.bottom = y
        else: rect.top = y
        glow_surf = pygame.Surface(rect.size, pygame.SRCALPHA); glow_surf.fill((0,0,0,0))
        glow_radius = 2 + 2*math.sin(self.glow_phase)
        for alpha_val in self.GLOW_ALPHA_VALUES:
            temp_text = font.render(text.upper(),True,(255,255,255,255)).convert_alpha(); temp_text.set_alpha(alpha_val)
            color_surf = pygame.Surface(temp_text.get_size(),pygame.SRCALPHA); color_surf.fill((*color,255))
            temp_text.blit(color_surf,(0,0),special_flags=pygame.BLEND_RGBA_MULT)
            for dx,dy in [(-glow_radius,0),(glow_radius,0),(0,-glow_radius),(0,glow_radius)]:
                glow_surf.blit(temp_text, (glow_surf.get_width()//2-temp_text.get_width()//2+int(dx), glow_surf.get_height()//2-temp_text.get_height()//2+int(dy)))
        glow_surf.blit(base_surf, (glow_surf.get_width()//2-base_surf.get_width()//2, glow_surf.get_height()//2-base_surf.get_height()//2))
        final_rect = glow_surf.get_rect()
        if "right" in align: final_rect.right=rect.right
        else: final_rect.left=rect.left
        if "bottom" in align: final_rect.bottom=rect.bottom
        else: final_rect.top=rect.top
        surface.blit(glow_surf, final_rect)

class FancyLootToken:
    def __init__(self, pos, color, value=10, particle_system=None):
        self.pos=Vector2(pos); self.vel=Vector2(random.uniform(-0.5,0.5),random.uniform(-0.5,0.5))
        self.color=color; self.value=value; self.radius=8; self.timer=0
        self.particle_system = particle_system
    def update(self):
        self.timer+=1; wave_mag=0.2; self.vel.y+=wave_mag*math.sin(self.timer*0.05); self.vel.x+=wave_mag*math.cos(self.timer*0.07)
        self.pos+=self.vel*0.3; self.vel*=0.98
    def draw(self, surface):
        angle=self.timer*0.1; num_spikes=5; points=[]
        for i in range(num_spikes*2):
            r_val = self.radius if i%2==0 else self.radius*0.5; theta = angle+(math.pi*2*i)/(num_spikes*2) # Renamed r
            points.append((self.pos.x+r_val*math.cos(theta), self.pos.y+r_val*math.sin(theta)))
        pygame.draw.polygon(surface, self.color, points)
    def check_collision(self, player): return (self.pos-player.pos).length_squared()<(self.radius+20)**2
    def on_collected(self):
        if self.particle_system: self.particle_system.spawn_star_burst(self.pos,color=self.color,count=10)

class LootToken:
    def __init__(self, pos, color, value=10):
        self.pos=Vector2(pos); self.vel=Vector2(random.uniform(-1,1),random.uniform(-1,1))
        self.color=color; self.value=value; self.radius=8
    def update(self): self.pos+=self.vel*0.3; self.vel*=0.98; self.pos=wrap_position(self.pos,WIDTH,HEIGHT)
    def draw(self, surface): pygame.draw.circle(surface,self.color,(int(self.pos.x),int(self.pos.y)),self.radius)
    def check_collision(self, player): return (self.pos-player.pos).length_squared()<(self.radius+20)**2

class TrailSegment: # TrailSegment class is still in main.py for now
    def __init__(self, pos, angle, color, lifetime=30):
        self.pos=Vector2(pos); self.angle=angle; self.color=color; self.lifetime=lifetime
    def update(self): self.lifetime-=1
    def draw(self, surface):
        alpha_ratio=self.lifetime/30; c=(int(self.color[0]*alpha_ratio),int(self.color[1]*alpha_ratio),int(self.color[2]*alpha_ratio))
        sp=[Vector2(15,0),Vector2(-8,8),Vector2(-8,-8)]; rad=math.radians(self.angle); ca=math.cos(rad); sa=math.sin(rad); pts=[]
        for p in sp: x=p.x*ca-p.y*sa; y=p.x*sa+p.y*ca; pts.append((x+self.pos.x,y+self.pos.y))
        pygame.draw.polygon(surface,c,pts)

class FancyCountdown:
    def __init__(self, fps=60, base_font_size=100, color=(255,255,255), font_name=None):
        self.fps=fps; self.base_font_size=base_font_size; self.color=color; self.font_name=font_name
        self.active=False; self.frames_left=0; self.total_seconds=0; self.seconds_left=0; self.last_seconds_left=-1
    def start(self, total_seconds=10):
        self.total_seconds=total_seconds; self.frames_left=total_seconds*self.fps; self.active=True; self.seconds_left=total_seconds; self.last_seconds_left=-1
    def stop(self): self.active=False; self.frames_left=0
    def update(self):
        if not self.active: return
        self.frames_left-=1
        if self.frames_left<0: self.frames_left=0; self.seconds_left=0; self.active=False
        else: self.seconds_left=(self.frames_left+(self.fps-1))//self.fps
        if self.seconds_left!=self.last_seconds_left: sfx_system.play_sound(SFX_COUNTDOWN); self.last_seconds_left=self.seconds_left
    def draw(self, surface):
        if not self.active or self.seconds_left<1 or self.seconds_left>self.total_seconds: return
        frame_in_current_second = self.frames_left%self.fps
        fraction_in_second = 0.0 if frame_in_current_second==0 else 1.0-(frame_in_current_second/float(self.fps))
        alpha=int(255*fraction_in_second); scale=2.0-fraction_in_second
        font_size=max(10,int(self.base_font_size*scale)); font_obj=pygame.font.SysFont(self.font_name,font_size)
        text_surf=font_obj.render(str(self.seconds_left),True,self.color).convert_alpha(); text_surf.set_alpha(alpha)
        surface.blit(text_surf,text_surf.get_rect(center=(surface.get_width()//2,surface.get_height()//2)))

class Room:
    def __init__(self, idx, is_boss=False):
        self.idx=idx; self.is_boss=is_boss; self.enemies=[]
        if not is_boss:
            for _ in range(random.randint(1,3)): self.enemies.append(Enemy((random.randint(100,WIDTH-100),random.randint(100,HEIGHT-100))))
        else: self.enemies.append(Enemy((WIDTH/4,HEIGHT/4),is_boss=True))
        self.loot=[]; self.particles=ParticleSystem()
        self.countdown=FancyCountdown(fps=FPS,base_font_size=120,color=(255,255,255))
    def start_collection(self): self.countdown.start(COLLECTION_TIME//1000 if COLLECTION_TIME>1000 else COLLECTION_TIME)
    def update(self, player, global_frame):
        for e in self.enemies: e.update(global_frame, self.enemies)
        for e in self.enemies[:]:
            for pb in player.bullets:
                if e.hit_by(pb):
                    e.health-=1; sfx_system.play_sound(SFX_HIT_ENEMY)
                    if pb.exploding:
                        for oe in self.enemies: 
                            if (oe.pos-pb.pos).length()<40: oe.health-=1
                    pb.life=0
                    if e.health<=0:
                        if not e.dropped_loot: e.dropped_loot=True; self.loot.append(FancyLootToken(e.pos,e.color,e.points,particle_system=self.particles)); player.score+=e.points
                        self.enemies.remove(e); break
        self.check_enemy_bullets_hit_player(player)
        for l in self.loot[:]:
            l.update()
            if l.check_collision(player): l.on_collected(); sfx_system.play_sound(27); player.score+=l.value*5; self.loot.remove(l) # SFX_COLLECT_LOOT or similar needed
        self.particles.update(); self.countdown.update()
    def check_enemy_bullets_hit_player(self, player):
        for e in self.enemies:
            for b in e.bullets[:]:
                if (b.pos-player.pos).length_squared()<(b.radius+20)**2: # Simplified hit check
                    if player.shield_hp>0: player.shield_hp-=1
                    else: player.hp-=1
                    sfx_system.play_sound(SFX_HIT_PLAYER); self.particles.spawn_hit_particles(b.pos,color=(255,200,50),count=8); e.bullets.remove(b)
    def draw(self, surface):
        for e in self.enemies: e.draw(surface)
        for l in self.loot: l.draw(surface)
        self.particles.draw(surface); self.countdown.draw(surface)
    def is_enemy_cleared(self): return len(self.enemies)==0
    def is_collection_done(self): return not self.countdown.active

g_pre_conversations = [[...],[...]] # Keep as is, too long to paste here
g_post_dialogs = [{...},{...}] # Keep as is

class DialogSystem:
    def __init__(self):
        self.pre_conversations=g_pre_conversations; self.post_dialogs=g_post_dialogs
        self.used_pre_conversations=set(); self.current_pre_conversation=[]; self.current_post_dialog=None
        self.state='idle'; self.timer=0; self.line_timer=0; self.DIALOG_TIME_PER_LINE=240; self.current_line=0
        self.branch_mode=False; self.selected_choice=None
    def is_pre_active(self): return self.state=='pre'
    def is_post_active(self): return self.state=='post'
    def start_post_dialog(self):
        if self.post_dialogs: self.current_post_dialog=random.choice(self.post_dialogs); self.branch_mode=True; self.selected_choice=None; self.state='post'; self.timer=0
        else: self.state='idle'
    def start_random_pre_dialog(self):
        available=[i for i in range(len(self.pre_conversations)) if i not in self.used_pre_conversations]
        if available: idx=random.choice(available); self.used_pre_conversations.add(idx); self.current_pre_conversation=self.pre_conversations[idx]; self.current_line=0; self.timer=0; self.line_timer=0; self.state='pre'
        else: self.current_pre_conversation=[]; self.state='idle'
    def next_line(self):
        self.current_line+=1; self.line_timer=0
        if self.current_line>=len(self.current_pre_conversation): self.state='idle'
    def update(self, keys, player):
        if self.state=='pre':
            self.line_timer+=1
            if keys[pygame.K_RETURN] or self.line_timer>self.DIALOG_TIME_PER_LINE: self.next_line()
        elif self.state=='post' and self.current_post_dialog:
            if self.branch_mode:
                if keys[pygame.K_1]: self.selected_choice=0; self.branch_mode=False; self.apply_choice_effect(0,player)
                elif keys[pygame.K_2]: self.selected_choice=1; self.branch_mode=False; self.apply_choice_effect(1,player)
            else: self.timer+=1; 
            if self.timer>self.DIALOG_TIME_PER_LINE: self.state='idle'
    def apply_choice_effect(self, choice_idx, player=None):
        if self.current_post_dialog and 'effects' in self.current_post_dialog and player:
            effects=self.current_post_dialog['effects'][choice_idx]
            if 'score' in effects: player.score+=effects['score']
            if 'repair' in effects: player.hp=player.max_hp
    def draw(self, surface):
        if self.state=='pre' and self.current_pre_conversation and self.current_line<len(self.current_pre_conversation):
            draw_text(surface,self.current_pre_conversation[self.current_line],WIDTH//2,HEIGHT//2-100,center=True)
        elif self.state=='post' and self.current_post_dialog:
            draw_text(surface,self.current_post_dialog['prompt'],WIDTH//2,HEIGHT//2-120,center=True)
            if self.branch_mode:
                y=HEIGHT//2-40
                for idx,(choice,_) in enumerate(self.current_post_dialog['choices']): draw_text(surface,choice,WIDTH//2,y,color=(0,255,255),center=True); y+=40
            elif self.selected_choice is not None: draw_text(surface,self.current_post_dialog['choices'][self.selected_choice][1],WIDTH//2,HEIGHT//2-40,center=True)
    def is_pre_done(self): return self.state!='pre'
    def is_post_done(self): return self.state!='post'

class StoreSystem:
    def __init__(self): self.active=False; self.message=""; self.timer=0; self.prev_keys=None; self.items_purchased=set()
    def open_store(self): self.active=True; self.message="Store: Press [1-9] to buy power-ups, ESC/ENTER to exit"; self.timer=0
    def close_store(self): self.active=False; self.message=""
    def update(self, player, keys):
        if self.prev_keys is None: self.prev_keys={k:keys[k] for k in [pygame.K_ESCAPE,pygame.K_RETURN,pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5,pygame.K_6,pygame.K_7,pygame.K_8,pygame.K_9]}; return
        if not self.active: return
        self.timer+=1
        if keys[pygame.K_ESCAPE] or keys[pygame.K_RETURN]: self.close_store(); return
        for i in range(1,10):
            key_code=getattr(pygame,f'K_{i}')
            if keys[key_code] and not self.prev_keys[key_code]: self.buy_item(player,i-1)
        for k in self.prev_keys: self.prev_keys[k]=keys[k]
    def buy_item(self, player, index):
        if not (0<=index<len(POWERUPS)) or index in self.items_purchased: return
        item=POWERUPS[index]
        if player.score>=item["cost"]:
            sfx_system.play_sound(SFX_BUY_SUCCESS); player.score-=item["cost"]; self.apply_powerup(player,index); self.items_purchased.add(index)
            self.message=f"Bought {item['name']} for {item['cost']} pts!"
    def apply_powerup(self,player,index):
        name=POWERUPS[index]["name"]
        if name=="Homing Bullets": player.upgrades["homing_bullet_level"]=max(player.upgrades["homing_bullet_level"],1)
        elif name=="Better Homing Bullets": player.upgrades["homing_bullet_level"]=2
        elif name=="Exploding Bullets": player.upgrades["exploding_bullets"]=True
        elif name=="Bigger Bullets": player.upgrades["bigger_bullets"]=True
        elif name=="Ship Shield": player.upgrades["shield"]=True; player.shield_max=30; player.shield_hp=player.shield_max
        elif name=="Ship Teleporter": player.upgrades["teleporter"]=True
        elif name=="Short-Range Autofire": player.upgrades["short_range_autofire"]=True
        elif name=="Mines": player.upgrades["mines"]=True
        elif name=="Homing Mines": player.upgrades["mines"]=True; player.upgrades["homing_mines"]=True
    def draw(self,surface):
        if self.active:
            draw_text(surface,self.message,WIDTH//2,HEIGHT//2-200,center=True)
            y=HEIGHT//2-140
            for i,p in enumerate(POWERUPS):
                color=(120,120,120) if i in self.items_purchased else (255,255,0)
                draw_text(surface,f"{i+1}) {p['name']} ({p['cost']} pts) - {p['desc']}",WIDTH//2,y,color=color,font_size=22,center=True); y+=30

class StarMapSystem:
    def __init__(self, num_rooms):
        self.active=False; self.num_rooms=num_rooms; self.node_positions=[]
        for i in range(num_rooms): self.node_positions.append((random.randint(50,450),random.randint(50,450)))
        self.edges=[]; unconnected=list(range(num_rooms)); random.shuffle(unconnected)
        while len(unconnected)>1: a=unconnected.pop(); b=unconnected[-1]; e=tuple(sorted([a,b])); 
        if e not in self.edges: self.edges.append(e)
        for _ in range(num_rooms): a=random.randint(0,num_rooms-1); b=random.randint(0,num_rooms-1); 
        if a!=b: e=tuple(sorted([a,b])); 
        if e not in self.edges: self.edges.append(e)
        self.MAP_TIME=240; self.timer=0; self.anim_time=0.0; self.prev_node=0; self.next_node=0; self.travel_alpha=0.0
    def open_map(self,prev_node_idx,next_node_idx): self.active=True; self.timer=0; self.anim_time=0.0; self.prev_node=prev_node_idx; self.next_node=next_node_idx; self.travel_alpha=0.0; sfx_system.play_sound(SFX_STARMAP_OPEN)
    def close_map(self): self.active=False; sfx_system.play_sound(SFX_STARMAP_CLOSE)
    def update(self,dt=1/60):
        if not self.active: return
        self.timer+=1; self.anim_time+=dt
        if self.timer>self.MAP_TIME: self.close_map()
        total_travel_frames=self.MAP_TIME*0.5
        if self.timer>self.MAP_TIME-total_travel_frames: self.travel_alpha=min(1.0,max(0.0,(self.timer-(self.MAP_TIME-total_travel_frames))/total_travel_frames))
        else: self.travel_alpha=0.0
    def draw(self,surface):
        if not self.active: return
        self.draw_dark_background(surface); self.draw_beveled_border(surface,200,100,500,500,corner_radius=30)
        self.draw_pulsing_edges(surface,200,100); self.draw_nodes_and_reticle(surface,200,100)
        label=pygame.font.SysFont(None,26).render("Star Map - Warping...",True,(255,255,255))
        surface.blit(label,label.get_rect(center=(200+250,100+30)))
    def draw_dark_background(self,surface):
        w,h=surface.get_size()
        for y_pos in range(h): ratio=y_pos/h; r_val,g_val,b_val=[int(c*(1-ratio)) for c in [10,10,20]]; pygame.draw.line(surface,(r_val,g_val,b_val),(0,y_pos),(w,y_pos)) # Renamed r,g,b
    def draw_beveled_border(self,surface,x,y,w,h,corner_radius=20):
        pygame.draw.rect(surface,(40,40,80),pygame.Rect(x,y,w,h),border_radius=corner_radius)
        pygame.draw.rect(surface,(100,100,150),pygame.Rect(x,y,w,h),width=4,border_radius=corner_radius)
    def draw_pulsing_edges(self,surface,map_x,map_y):
        for(a,b) in self.edges:
            x1,y1=self.node_positions[a]; x2,y2=self.node_positions[b]; pulse=0.5+0.5*math.sin(self.anim_time*2+a+b)
            edge_color=(int(255*pulse),int(120*pulse),int(255*(1-pulse)))
            pygame.draw.line(surface,edge_color,(map_x+x1,map_y+y1),(map_x+x2,map_y+y2),2)
    def draw_nodes_and_reticle(self,surface,map_x,map_y):
        for i,(nx,ny) in enumerate(self.node_positions):
            px,py=map_x+nx,map_y+ny
            c=(0,255,0) if i==self.prev_node else (255,165,0) if i==self.next_node else (255,255,255)
            pygame.draw.circle(surface,c,(px,py),8)
            ring_radius=12+4*math.sin(self.anim_time*4+i); ring_color=tuple(min(255,val+50) for val in c)
            pygame.draw.circle(surface,ring_color,(px,py),max(1,int(ring_radius)),width=1)
        if 0<=self.travel_alpha<=1.0:
            (prev_x,prev_y)=self.node_positions[self.prev_node]; (next_x,next_y)=self.node_positions[self.next_node]
            cx=prev_x+(next_x-prev_x)*self.travel_alpha; cy=prev_y+(next_y-prev_y)*self.travel_alpha
            px,py=map_x+cx,map_y+cy; ret_size=20+10*math.sin(self.anim_time*2); ret_color=(255,255,0)
            pygame.draw.line(surface,ret_color,(px-ret_size,py),(px+ret_size,py),2); pygame.draw.line(surface,ret_color,(px,py-ret_size),(px,py+ret_size),2)
            pygame.draw.circle(surface,ret_color,(int(px),int(py)),int(ret_size),width=1)

class Star:
    def __init__(self, layer, player_ship):
        self.x=random.uniform(0,WIDTH); self.y=random.uniform(0,HEIGHT); self.size=random.uniform(1,3)
        self.brightness=random.randint(50,255); self.layer=layer; self.speed_multiplier=0.2+(layer*0.2)
        self.color=(self.brightness,)*3; self.player_ship=player_ship
    def update(self):
        self.x-=self.player_ship.vel.x*self.speed_multiplier; self.y-=self.player_ship.vel.y*self.speed_multiplier
        if self.x<0:self.x+=WIDTH elif self.x>=WIDTH:self.x-=WIDTH
        if self.y<0:self.y+=HEIGHT elif self.y>=HEIGHT:self.y-=HEIGHT
    def draw(self,surface):pygame.draw.circle(surface,self.color,(int(self.x),int(self.y)),int(self.size))

class Starfield:
    def __init__(self,num_stars,player_ship):
        self.stars=[]; self.player_ship=player_ship
        for layer in range(3):
            for _ in range(num_stars//3): self.stars.append(Star(layer,player_ship))
    def update(self): 
        for star in self.stars: star.update()
    def draw(self,surface): 
        for star in self.stars: star.draw(surface)

sfx_system = None 


async def main():
    global sfx_system 
    pygame.init()
    
    sfx_system = SfxSystem() 

    FinalFlashingPressStart().run()

    if MUSIC_ENABLED: 
        pygame.mixer.music.load("assets/cq1.wav") 
        pygame.mixer.music.play(-1)        

    screen = pygame.display.set_mode((WIDTH, HEIGHT)) 
    pygame.display.set_caption("WedgeRogue with HP & Fancy Loot")
    clock = pygame.time.Clock()
    player = PlayerShip() # PlayerShip is now imported
    starfield = Starfield(100, player)
    rooms=[]
    for i in range(NUM_ROOMS): 
        rooms.append(Room(i, is_boss=(i==NUM_ROOMS-1))) 
    dialog = DialogSystem()
    store  = StoreSystem()
    star_map = StarMapSystem(NUM_ROOMS) 
    hud = HUDSystem()
    options_menu=OptionsMenu()
    current_room_idx = 0
    game_state = STATE_PRE_DIALOG 
    dialog.start_random_pre_dialog()
    global_frame = 0
    running = True
    while running:
        dt = clock.tick(FPS) 
        global_frame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if game_state not in (STATE_STORE, STATE_STAR_MAP): running = False
                elif event.key == pygame.K_SPACE:
                    if game_state in (STATE_COMBAT, STATE_COLLECTION): sfx_system.play_sound(SFX_SHOOT if isinstance(SFX_SHOOT, int) else 10); player.shoot() 
                elif event.key==pygame.K_o:
                    if not options_menu.active: options_menu.open()
                    else: options_menu.close()
        
        current_screen_shake_mag = SCREEN_SHAKE_MAG 
        
        if game_state == STATE_END: keys = pygame.key.get_pressed() 
        else:
            keys = pygame.key.get_pressed()
            star_map.update() 
            store.update(player, keys)
            if dialog.is_pre_active(): dialog.update(keys, player)
            elif dialog.is_post_active(): dialog.update(keys, player)

            if options_menu.active: options_menu.update(keys)
            elif game_state == STATE_PRE_DIALOG:
                if dialog.is_pre_done(): game_state = STATE_COMBAT
            elif game_state == STATE_COMBAT:
                current_room = rooms[current_room_idx]; starfield.update(); player.update(keys, current_room.enemies); current_room.update(player, global_frame)
                if player.hp <= 0: game_state = STATE_END
                elif current_room.is_enemy_cleared():
                    if not current_room.countdown.active: current_room.start_collection()
                    game_state = STATE_COLLECTION
            elif game_state == STATE_COLLECTION:
                current_room = rooms[current_room_idx]; starfield.update(); player.update(keys, current_room.enemies); current_room.update(player, global_frame)
                if player.hp <= 0: game_state = STATE_END
                elif current_room.is_collection_done(): dialog.start_post_dialog(); game_state = STATE_POST_DIALOG
            elif game_state == STATE_POST_DIALOG:
                if dialog.is_post_done(): store.open_store(); game_state = STATE_STORE
            elif game_state == STATE_STORE:
                if not store.active:
                    prev_idx = current_room_idx; next_idx = (current_room_idx + 1) % NUM_ROOMS
                    star_map.open_map(prev_idx, next_idx); game_state = STATE_STAR_MAP; player.reset()
            elif game_state == STATE_STAR_MAP:
                if not star_map.active:
                    current_room_idx = (current_room_idx + 1) % NUM_ROOMS
                    dialog.start_random_pre_dialog(); game_state = STATE_PRE_DIALOG
        
        screen.fill((0,0,0))
        
        render_surface = screen
        if SCREEN_SHAKE_MAG > 0: 
            pass 

        if game_state == STATE_STAR_MAP: star_map.draw(render_surface)
        else:
            starfield.draw(render_surface); cur_room = rooms[current_room_idx]; cur_room.draw(render_surface)
            player.draw(render_surface); dialog.draw(render_surface); store.draw(render_surface); hud.draw(render_surface, player, WIDTH, HEIGHT)
            draw_instructions(render_surface, game_state, cur_room) 
        if game_state == STATE_END:
            draw_text(render_surface,"GAME OVER!",WIDTH//2,HEIGHT//2,color=(255,50,50),font_size=48,center=True) 
            draw_text(render_surface,"Press ESC to Quit",WIDTH//2,HEIGHT//2+60,color=(255,255,255),font_size=32,center=True) 
        
        pygame.display.flip()

    if os.environ.get('PYGBAG') is None: pygame.quit(); sys.exit()

asyncio.run(main())

[end of main.py]
