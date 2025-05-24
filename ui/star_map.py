import pygame
import random
import math
from sfx import sfx_system # For sound effects
from config import SFX_STARMAP_OPEN, SFX_STARMAP_CLOSE # Specific SFX IDs

class StarMapSystem:
    def __init__(self, num_rooms):
        self.active = False
        self.num_rooms = num_rooms # Number of nodes/rooms on the map

        # Generate node positions randomly within a defined map area
        # These could be fixed or based on a more complex generation algorithm
        self.node_positions = []
        for _ in range(num_rooms):
            # Assuming map display area is somewhat fixed or relative
            nx = random.randint(50, 450) # Example coordinates
            ny = random.randint(50, 450)
            self.node_positions.append((nx, ny))

        # Generate edges to connect nodes, ensuring a connected graph
        self.edges = []
        if num_rooms > 0: # Ensure there are rooms to connect
            unconnected = list(range(num_rooms))
            random.shuffle(unconnected)
            # Create a path through all nodes to ensure connectivity
            for i in range(len(unconnected) - 1):
                a, b = unconnected[i], unconnected[i+1]
                edge = tuple(sorted((a, b)))
                if edge not in self.edges:
                    self.edges.append(edge)
            
            # Add a few more random edges for more connections (optional)
            # The number of extra edges can be a fraction of num_rooms or a fixed number
            for _ in range(num_rooms // 2): 
                a = random.randint(0, num_rooms - 1)
                b = random.randint(0, num_rooms - 1)
                if a != b:
                    edge = tuple(sorted((a, b)))
                    if edge not in self.edges:
                        self.edges.append(edge)
        
        self.MAP_TIME = 240    # Duration the map is shown (in frames)
        self.timer = 0         # Current frame count for map display time
        self.anim_time = 0.0   # For continuous animations like pulsing

        # For animating the "traveling highlight" or reticle
        self.prev_node = 0
        self.next_node = 0
        self.travel_alpha = 0.0  # Interpolation factor (0.0 to 1.0)

    def open_map(self, prev_node_idx, next_node_idx):
        self.active = True
        self.timer = 0
        self.anim_time = 0.0
        self.prev_node = prev_node_idx
        self.next_node = next_node_idx
        self.travel_alpha = 0.0
        if sfx_system: sfx_system.play_sound(SFX_STARMAP_OPEN)

    def close_map(self):
        self.active = False
        if sfx_system: sfx_system.play_sound(SFX_STARMAP_CLOSE)
 
    def update(self, dt=1/60.0): # dt is delta time, default for 60 FPS
        if not self.active:
            return
        
        self.timer += 1
        self.anim_time += dt

        if self.timer > self.MAP_TIME:
            self.close_map()
            return # Important to return after closing to avoid further updates this frame

        # Animate travel_alpha (e.g., over the latter half of MAP_TIME)
        travel_duration_frames = self.MAP_TIME * 0.5 # Example: travel animation takes half the map time
        travel_start_frame = self.MAP_TIME - travel_duration_frames
        
        if self.timer > travel_start_frame:
            progress = (self.timer - travel_start_frame) / travel_duration_frames
            self.travel_alpha = min(1.0, max(0.0, progress)) # Clamp between 0 and 1
        else:
            self.travel_alpha = 0.0

    def draw(self, surface):
        if not self.active:
            return

        # Define map area (these could be passed in or configured)
        map_rect_x, map_rect_y, map_rect_w, map_rect_h = 150, 80, 600, 500 # Slightly adjusted
        corner_radius = 30
        
        self.draw_dark_background(surface)
        self.draw_beveled_border(surface, map_rect_x, map_rect_y, map_rect_w, map_rect_h, corner_radius)
        
        # Adjust node positions to be relative to the map_rect for drawing
        # For now, assuming node_positions are already scaled/positioned for the hardcoded map_rect_x/y offsets
        self.draw_pulsing_edges(surface, map_rect_x, map_rect_y) 
        self.draw_nodes_and_reticle(surface, map_rect_x, map_rect_y)

        # Draw a title label for the star map
        try:
            font = pygame.font.SysFont(None, 28) # Slightly larger font
            label = font.render("INTERSTELLAR MAP - CALCULATING ROUTE...", True, (220, 220, 255)) # Light blueish text
            label_rect = label.get_rect(center=(map_rect_x + map_rect_w // 2, map_rect_y + 30))
            surface.blit(label, label_rect)
        except Exception as e: # Catch font rendering errors
            print(f"Error rendering star map label: {e}")


    def draw_dark_background(self, surface):
        w, h = surface.get_size()
        # Simple vertical gradient from dark blue/purple to black
        top_color = (10, 5, 25)
        bottom_color = (0, 0, 0)
        for y_pos in range(h):
            ratio = y_pos / float(h)
            r = int(top_color[0] * (1 - ratio) + bottom_color[0] * ratio)
            g = int(top_color[1] * (1 - ratio) + bottom_color[1] * ratio)
            b = int(top_color[2] * (1 - ratio) + bottom_color[2] * ratio)
            pygame.draw.line(surface, (r, g, b), (0, y_pos), (w, y_pos))

    def draw_beveled_border(self, surface, x, y, w, h, corner_radius):
        rect_color = (20, 20, 50, 200)  # Dark blue with some alpha
        border_color = (80, 80, 150, 220) # Lighter blue/purple border
        
        # Create a temporary surface for transparency
        temp_surface = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(temp_surface, rect_color, temp_surface.get_rect(), border_radius=corner_radius)
        pygame.draw.rect(temp_surface, border_color, temp_surface.get_rect(), width=3, border_radius=corner_radius) # Border
        surface.blit(temp_surface, (x,y))


    def draw_pulsing_edges(self, surface, map_origin_x, map_origin_y):
        for (a_idx, b_idx) in self.edges:
            pos_a = self.node_positions[a_idx]
            pos_b = self.node_positions[b_idx]
            
            # Pulsation effect for edge color
            pulse = 0.6 + 0.4 * math.sin(self.anim_time * 3 + a_idx + b_idx) # Adjusted speed and base
            edge_color = (
                int(max(0, min(255, 100 * pulse))),  # Dimmer base, brighter pulse for blue/purple
                int(max(0, min(255, 100 * pulse))),
                int(max(0, min(255, 200 * pulse + 55))) 
            )
            
            start_pos = (map_origin_x + pos_a[0], map_origin_y + pos_a[1])
            end_pos = (map_origin_x + pos_b[0], map_origin_y + pos_b[1])
            pygame.draw.line(surface, edge_color, start_pos, end_pos, 2)

    def draw_nodes_and_reticle(self, surface, map_origin_x, map_origin_y):
        node_radius = 7
        reticle_base_size = 15

        for i, (nx, ny) in enumerate(self.node_positions):
            px = map_origin_x + nx
            py = map_origin_y + ny

            node_color = (180, 180, 220) # Default node color (light lavender)
            if i == self.prev_node:
                node_color = (0, 255, 0)  # Green for previous/current node
            elif i == self.next_node:
                node_color = (255, 165, 0) # Orange for next destination node

            pygame.draw.circle(surface, node_color, (px, py), node_radius)
            
            # Flicker/pulse effect for node outline
            ring_pulse = 0.7 + 0.3 * math.sin(self.anim_time * 5 + i)
            ring_radius_val = node_radius + 3 + 2 * ring_pulse
            ring_color_val = (min(255,node_color[0]+40), min(255,node_color[1]+40), min(255,node_color[2]+40))
            pygame.draw.circle(surface, ring_color_val, (px, py), int(ring_radius_val), width=1)

        # Draw the animated reticle traveling between nodes
        if 0 <= self.travel_alpha <= 1.0 and self.prev_node < len(self.node_positions) and self.next_node < len(self.node_positions):
            prev_pos_x, prev_pos_y = self.node_positions[self.prev_node]
            next_pos_x, next_pos_y = self.node_positions[self.next_node]

            # Interpolate reticle position
            reticle_x = map_origin_x + (prev_pos_x + (next_pos_x - prev_pos_x) * self.travel_alpha)
            reticle_y = map_origin_y + (prev_pos_y + (next_pos_y - prev_pos_y) * self.travel_alpha)
            
            # Dynamic reticle size (e.g., pulsing)
            current_ret_size = reticle_base_size + 5 * math.sin(self.anim_time * 2.5)
            ret_color = (255, 255, 0) # Yellow

            # Draw crosshair lines for reticle
            pygame.draw.line(surface, ret_color, (reticle_x - current_ret_size, reticle_y), (reticle_x + current_ret_size, reticle_y), 2)
            pygame.draw.line(surface, ret_color, (reticle_x, reticle_y - current_ret_size), (reticle_x, reticle_y + current_ret_size), 2)
            # Draw a circle for the reticle center
            pygame.draw.circle(surface, ret_color, (int(reticle_x), int(reticle_y)), int(current_ret_size * 0.7), width=1)
