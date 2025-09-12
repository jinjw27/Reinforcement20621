import pygame
import random
import math
import sys

# ----------------------
# Initialization
# ----------------------
pygame.init()
pygame.font.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fluid Horizon")

# Colors
BG_COLOR = (240, 244, 248)       # Soft pastel background
SHIP_COLOR = (78, 205, 196)      # Vibrant accent for ship
OBSTACLE_COLOR = (255, 107, 107) # Vibrant accent for obstacles
PARTICLE_COLOR = (78, 205, 196)
TEXT_COLOR = (50, 50, 50)

# Fonts
FONT = pygame.font.SysFont("Arial", 36)
SMALL_FONT = pygame.font.SysFont("Arial", 24)

# Game constants
FPS = 60
GRAVITY = 0.5
THRUST = -10
OBSTACLE_SPEED = 5
OBSTACLE_INTERVAL = 1500  # milliseconds
OBSTACLE_WIDTH = 80
GAP_HEIGHT = 200

# ----------------------
# Game Classes
# ----------------------
class Ship:
    def __init__(self):
        self.width = 40
        self.height = 30
        self.x = 150
        self.y = HEIGHT // 2
        self.vel_y = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.particles = []

    def update(self, keys):
        # Controls
        if keys[pygame.K_SPACE]:
            self.vel_y = THRUST

        # Physics
        self.vel_y += GRAVITY
        self.y += self.vel_y
        self.rect.y = int(self.y)

        # Keep ship in screen bounds
        if self.y < 0:
            self.y = 0
            self.vel_y = 0
        if self.y + self.height > HEIGHT:
            self.y = HEIGHT - self.height
            self.vel_y = 0

        # Particles for thrust
        if keys[pygame.K_SPACE]:
            self.particles.append(Particle(self.x, self.y + self.height // 2))

        # Update particles
        for p in self.particles[:]:
            p.update()
            if p.lifetime <= 0:
                self.particles.remove(p)

    def draw(self, win):
        # Draw particles behind ship
        for p in self.particles:
            p.draw(win)
        # Draw ship (triangle)
        pygame.draw.polygon(
            win,
            SHIP_COLOR,
            [
                (self.x, self.y + self.height // 2),
                (self.x + self.width, self.y),
                (self.x + self.width, self.y + self.height),
            ],
        )

class Obstacle:
    def __init__(self, x):
        self.width = OBSTACLE_WIDTH
        self.gap_y = random.randint(100, HEIGHT - 100 - GAP_HEIGHT)
        self.x = x
        self.passed = False
        # Top and bottom rectangles
        self.top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y)
        self.bottom_rect = pygame.Rect(self.x, self.gap_y + GAP_HEIGHT, self.width, HEIGHT - self.gap_y - GAP_HEIGHT)

    def update(self):
        self.x -= OBSTACLE_SPEED
        self.top_rect.x = int(self.x)
        self.bottom_rect.x = int(self.x)

    def draw(self, win):
        pygame.draw.rect(win, OBSTACLE_COLOR, self.top_rect, border_radius=10)
        pygame.draw.rect(win, OBSTACLE_COLOR, self.bottom_rect, border_radius=10)

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = random.randint(3, 6)
        self.vel_x = random.uniform(-1, -3)
        self.vel_y = random.uniform(-1, 1)
        self.lifetime = random.randint(15, 25)

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        self.lifetime -= 1
        self.radius *= 0.95

    def draw(self, win):
        if self.lifetime > 0:
            pygame.draw.circle(win, PARTICLE_COLOR, (int(self.x), int(self.y)), int(self.radius))

# ----------------------
# Game Functions
# ----------------------
def draw_window(win, ship, obstacles, score):
    win.fill(BG_COLOR)
    ship.draw(win)
    for obs in obstacles:
        obs.draw(win)

    # Score
    score_text = SMALL_FONT.render(f"Score: {score}", True, TEXT_COLOR)
    win.blit(score_text, (10, 10))

    pygame.display.update()

def check_collision(ship, obstacles):
    for obs in obstacles:
        if ship.rect.colliderect(obs.top_rect) or ship.rect.colliderect(obs.bottom_rect):
            return True
    return False

def main():
    clock = pygame.time.Clock()
    ship = Ship()
    obstacles = []
    score = 0
    run = True
    last_obstacle = pygame.time.get_ticks()

    while run:
        clock.tick(FPS)
        keys = pygame.key.get_pressed()

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update ship
        ship.update(keys)

        # Spawn obstacles
        now = pygame.time.get_ticks()
        if now - last_obstacle > OBSTACLE_INTERVAL:
            obstacles.append(Obstacle(WIDTH))
            last_obstacle = now

        # Update obstacles
        for obs in obstacles[:]:
            obs.update()
            if obs.x + obs.width < 0:
                obstacles.remove(obs)
                score += 1

        # Check collisions
        if check_collision(ship, obstacles):
            run = False

        draw_window(WIN, ship, obstacles, score)

    # Game Over Screen
    WIN.fill(BG_COLOR)
    game_over_text = FONT.render("GAME OVER", True, TEXT_COLOR)
    score_text = SMALL_FONT.render(f"Score: {score}", True, TEXT_COLOR)
    WIN.blit(game_over_text, (WIDTH//2 - game_over_text.get_width()//2, HEIGHT//2 - 50))
    WIN.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//2 + 10))
    pygame.display.update()
    pygame.time.delay(3000)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
