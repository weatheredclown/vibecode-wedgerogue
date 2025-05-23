import pygame
from pygame.math import Vector2
import random
import math
from utils import wrap_position, trigger_shake
from .bullet import Bullet # Use the actual Bullet class from .bullet
from sfx import sfx_system 
from config import WIDTH, HEIGHT, BULLET_INTERVAL

ENEMY_TYPES = [
    ((255,   0,   0), 5,  5),
    ((255, 140,   0), 8,  10),
    ((255, 255,   0), 12, 15),
    ((  0, 255,   0), 15, 20),
    ((  0, 200, 200),20, 25)
]

class Enemy:
    def __init__(self, pos, is_boss=False):
        self.pos = Vector2(pos)
        self.radius = 20
        self.vel = Vector2(random.uniform(-2, 2), random.uniform(-2, 2))
        self.bullets = [] 
        self.angle = 0
        self.timer = 0

        etype = random.choice(ENEMY_TYPES)
        self.color = etype[0]
        self.max_health = etype[1]
        self.health = self.max_health
        self.points = etype[2]
        self.dropped_loot = False

        self.is_boss = is_boss
        if self.is_boss:
            self.max_health = 200
            self.health = self.max_health
            self.radius = 30
            self.color = (255, 80, 80)

        try:
            self.sprite = pygame.image.load("assets/enemy.png").convert_alpha()
            self.sprite_height = 40
            sprite_width = int(self.sprite.get_width() * (self.sprite_height / self.sprite.get_height()))
            self.sprite = pygame.transform.scale(self.sprite, (sprite_width, self.sprite_height))
        except pygame.error as e:
            print(f"Warning: Could not load enemy sprite: {e}")
            self.sprite = None


    def update(self, global_frame, enemies_list_for_homing_unused): # Renamed to clarify it's not used for enemy's own bullets
        self.pos += self.vel * 0.5
        self.pos = wrap_position(self.pos, WIDTH, HEIGHT)
        self.timer += 1

        if self.timer % BULLET_INTERVAL == 0:
            bullet_step = 30 if not self.is_boss else 15
            for angle_deg in range(0, 360, bullet_step):
                angle_rad = math.radians(angle_deg + self.timer * 5)
                bx = math.cos(angle_rad)
                by = math.sin(angle_rad)
                speed = 3 + 1 * math.sin(global_frame / 20)
                
                if sfx_system:
                    sfx_system.play_sound(10) 
                
                # Enemy bullets typically don't home on other enemies, so pass None or empty list for 'enemies' arg to Bullet.update
                self.bullets.append(Bullet(self.pos, Vector2(bx * speed, by * speed), color=self.color))

        for b in self.bullets:
            b.update(None) # Enemy bullets don't home on other enemies
        self.bullets = [b for b in self.bullets if not b.is_dead()]

    def draw(self, surface):
        diamond_points = self.get_diamond_points()
        pygame.draw.polygon(surface, (255, 255, 255), diamond_points, width=2)

        fraction = max(0.0, min(1.0, self.health / self.max_health))
        if fraction > 0:
            fill_poly = self.get_health_fill_polygon(fraction)
            pygame.draw.polygon(surface, self.color, fill_poly)

        for b in self.bullets:
            b.draw(surface)

        if self.sprite:
            rotated_sprite = pygame.transform.rotate(self.sprite, -self.angle)
            sprite_rect = rotated_sprite.get_rect(center=(self.pos.x, self.pos.y))
            surface.blit(rotated_sprite, sprite_rect.topleft)

    def get_diamond_points(self):
        x, y = self.pos; r = self.radius
        return [(x,y-r), (x+r,y), (x,y+r), (x-r,y)]

    def get_health_fill_polygon(self, fraction):
        diamond = self.get_diamond_points()
        top, right, bottom, left = diamond[0], diamond[1], diamond[2], diamond[3]
        fill_y = bottom[1] + fraction * (top[1] - bottom[1])
        edges = [(diamond[0],diamond[1]),(diamond[1],diamond[2]),(diamond[2],diamond[3]),(diamond[3],diamond[0])]
        fill_points = []
        for p1, p2 in edges:
            if p1[1] >= fill_y: fill_points.append(p1)
            cross = self.edge_intersects_horizontal(p1, p2, fill_y)
            if cross: fill_points.append(cross)
        
        cx, cy = self.pos
        unique_pts = list(dict.fromkeys(fill_points))
        unique_pts.sort(key=lambda pt: math.atan2(pt[1]-cy, pt[0]-cx))
        return unique_pts

    def edge_intersects_horizontal(self, p1, p2, line_y):
        x1,y1=p1; x2,y2=p2
        if (y1<line_y and y2<line_y) or (y1>line_y and y2>line_y) or abs(y2-y1)<1e-9: return None
        t=(line_y-y1)/(y2-y1)
        if 0<=t<=1: return (x1+t*(x2-x1), line_y)
        return None

    def hit_by(self, bullet): 
        dist_sq = (self.pos - bullet.pos).length_squared()
        if dist_sq < (self.radius + bullet.radius)**2:
            if bullet.exploding and bullet.bigger: 
                trigger_shake(15) 
            return True
        return False
