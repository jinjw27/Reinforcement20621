import pygame
import random
import math
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ----------------------
# Initialization
# ----------------------
pygame.init()
pygame.font.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600

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
    """Represents the player's ship with its physics properties."""
    def __init__(self):
        self.width = 40
        self.height = 30
        self.x = 100
        self.y = HEIGHT // 2
        self.velocity = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.color = SHIP_COLOR

    def update(self, thrust_on):
        """Applies physics (gravity and thrust) to the ship."""
        if thrust_on:
            self.velocity += THRUST
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = self.y
        # Clamp position to prevent going off-screen
        self.rect.y = max(0, min(self.rect.y, HEIGHT - self.rect.height))

    def draw(self, surface):
        """Draws the ship on the given surface."""
        pygame.draw.rect(surface, self.color, self.rect)

class Obstacle:
    """Represents a pair of top and bottom pipes with a gap."""
    def __init__(self, x):
        self.x = x
        self.width = OBSTACLE_WIDTH
        self.gap_height = GAP_HEIGHT
        # Randomly determine the center of the gap
        self.gap_center_y = random.randint(self.gap_height // 2, HEIGHT - self.gap_height // 2)

        # Create the top and bottom rectangles
        self.top_rect = pygame.Rect(self.x, 0, self.width, self.gap_center_y - self.gap_height // 2)
        self.bottom_rect = pygame.Rect(self.x, self.gap_center_y + self.gap_height // 2, self.width, HEIGHT)
        self.passed = False  # ✅ 중복 카운트 방지용 플래그 추가

    def update(self):
        """Moves the obstacle to the left."""
        self.x -= OBSTACLE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, surface):
        """Draws the obstacle on the given surface."""
        pygame.draw.rect(surface, OBSTACLE_COLOR, self.top_rect)
        pygame.draw.rect(surface, OBSTACLE_COLOR, self.bottom_rect)

def check_collision(ship, obstacles):
    """Checks for collision with any obstacle or the screen boundaries."""
    if ship.rect.top <= 0 or ship.rect.bottom >= HEIGHT:
        return True
    for obs in obstacles:
        if ship.rect.colliderect(obs.top_rect) or ship.rect.colliderect(obs.bottom_rect):
            return True
    return False

# ---------------------------------------------
# Gymnasium Environment Wrapper
# ---------------------------------------------
class FluidHorizonEnv(gym.Env):
    """A Gymnasium environment for the Fluid Horizon game."""
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        # Observation space: [ship y, ship vy, dist to obs, gap y, gap height]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)  # 0 = no thrust, 1 = thrust

        # Internal state
        self.ship = None
        self.obstacles = []
        self.score = 0
        self.passed_obstacles = 0

    def _get_obs(self):
        """Creates and normalizes the observation vector."""
        next_obstacle = None
        for obs in self.obstacles:
            if obs.x + obs.width > self.ship.x:
                next_obstacle = obs
                break

        if next_obstacle is None:
            rel_obs_x = 1.0
            obs_gap_y = 0.0
            obs_gap_h = 0.0
        else:
            rel_obs_x = (next_obstacle.x - self.ship.x) / WIDTH
            obs_gap_y = (next_obstacle.gap_center_y - (HEIGHT // 2)) / (HEIGHT // 2)
            obs_gap_h = next_obstacle.gap_height / HEIGHT

        normalized_velocity = self.ship.velocity / abs(THRUST * 1.5) if abs(THRUST) > 0 else 0

        obs = np.array([
            (self.ship.y - (HEIGHT // 2)) / (HEIGHT // 2),
            normalized_velocity,
            rel_obs_x,
            obs_gap_y,
            obs_gap_h
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        return {"score": self.score, "passed_obstacles": self.passed_obstacles}

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        self.ship = Ship()
        self.obstacles = []
        self.score = 0
        self.passed_obstacles = 0

        # Create initial obstacles
        for i in range(5):
            self.obstacles.append(Obstacle(WIDTH + i * (WIDTH // 2)))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        """Performs one step in the environment given an action."""
        terminated = False
        reward = 0.0

        # Update ship
        self.ship.update(action == 1)

        # Update and check obstacles
        for obs in self.obstacles[:]:
            obs.update()
            if obs.x + obs.width < self.ship.x and not obs.passed:
                obs.passed = True                     # ✅ 중복 방지
                self.passed_obstacles += 1
                self.score += 1                       # ✅ 점수 업데이트
                reward += 25.0

        # Remove obstacles off screen & add new
        if self.obstacles and self.obstacles[0].x + self.obstacles[0].width < 0:
            self.obstacles.pop(0)
            self.obstacles.append(Obstacle(self.obstacles[-1].x + WIDTH // 2))

        # Collision / survival
        if check_collision(self.ship, self.obstacles):
            reward = -100.0
            terminated = True
        else:
            reward += 0.1

        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Renders the game state to the screen."""
        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("Fluid Horizon RL")
                self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            self.screen.fill(BG_COLOR)

            self.ship.draw(self.screen)
            for obs in self.obstacles:
                obs.draw(self.screen)

            # ✅ 점수 표시
            score_text = FONT.render(f"Score: {int(self.score)}", True, TEXT_COLOR)
            self.screen.blit(score_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Closes the Pygame window."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# ---------------------------------------------
# Main execution for testing
# ---------------------------------------------
if __name__ == "__main__":
    env = FluidHorizonEnv(render_mode="human")
    observation, info = env.reset()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode finished. Final Score: {info['score']}")
            observation, info = env.reset()

    env.close()
