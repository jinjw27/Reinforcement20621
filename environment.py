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
BG_COLOR = (240, 244, 248)
SHIP_COLOR = (78, 205, 196)
OBSTACLE_COLOR = (255, 107, 107)
PARTICLE_COLOR = (78, 205, 196)
TEXT_COLOR = (50, 50, 50)

# Fonts
FONT = pygame.font.SysFont("Arial", 36)
SMALL_FONT = pygame.font.SysFont("Arial", 24)

# ----------------------
# Game constants
# ----------------------
FPS = 60
GRAVITY = 0.35
THRUST = -12
OBSTACLE_SPEED = 5
OBSTACLE_INTERVAL = 1500
OBSTACLE_WIDTH = 40
GAP_HEIGHT = 300

# ----------------------
# Game Classes
# ----------------------
class Ship:
    def __init__(self):
        self.width = 40
        self.height = 30
        self.x = 100
        self.color = SHIP_COLOR
        self.reset()
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def reset(self):
        # 중앙 대신 랜덤 y 위치에서 시작
        self.y = random.randint(HEIGHT // 4, 3 * HEIGHT // 4)
        self.velocity = 0
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, thrust_on):
        if thrust_on:
            self.velocity = THRUST
        self.velocity += GRAVITY
        self.y += self.velocity
        self.rect.y = max(0, min(self.y, HEIGHT - self.height))

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

class Obstacle:
    def __init__(self, x):
        self.x = x
        self.width = OBSTACLE_WIDTH
        self.gap_height = GAP_HEIGHT
        self.gap_center_y = random.randint(self.gap_height // 2, HEIGHT - self.gap_height // 2)
        self.top_rect = pygame.Rect(self.x, 0, self.width, self.gap_center_y - self.gap_height // 2)
        self.bottom_rect = pygame.Rect(self.x, self.gap_center_y + self.gap_height // 2, self.width, HEIGHT)
        self.passed = False

    def update(self):
        self.x -= OBSTACLE_SPEED
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, surface):
        pygame.draw.rect(surface, OBSTACLE_COLOR, self.top_rect)
        pygame.draw.rect(surface, OBSTACLE_COLOR, self.bottom_rect)

def check_collision(ship, obstacles):
    if ship.rect.top <= 0 or ship.rect.bottom >= HEIGHT:
        return True
    for obs in obstacles:
        if ship.rect.colliderect(obs.top_rect) or ship.rect.colliderect(obs.bottom_rect):
            return True
    return False

# ----------------------
# Gymnasium Environment Wrapper
# ----------------------
class FluidHorizonEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.ship = None
        self.obstacles = []
        self.score = 0
        self.passed_obstacles = 0

    def _get_obs(self):
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
        super().reset(seed=seed)

        self.ship = Ship()
        self.obstacles = []

        # 첫 번째 장애물 중앙 근처 생성
        self.obstacles.append(Obstacle(WIDTH))
        self.obstacles[0].gap_center_y = HEIGHT // 2

        # 나머지 장애물 랜덤 생성
        for i in range(1, 5):
            self.obstacles.append(Obstacle(WIDTH + i * (WIDTH // 2)))

        self.score = 0
        self.passed_obstacles = 0

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):
        terminated = False
        reward = 0.0

        self.ship.update(action == 1)

        for obs in self.obstacles[:]:
            obs.update()
            if obs.x + obs.width < self.ship.x and not obs.passed:
                obs.passed = True
                self.passed_obstacles += 1
                self.score += 1
                reward += 10.0
                if self.score % 5 == 0:
                    reward += 15.0

        if self.obstacles and self.obstacles[0].x + self.obstacles[0].width < 0:
            self.obstacles.pop(0)
            self.obstacles.append(Obstacle(self.obstacles[-1].x + WIDTH // 2))

        if check_collision(self.ship, self.obstacles):
            reward = -100.0
            terminated = True
        else:
            reward += 1

        observation = self._get_obs()
        info = self._get_info()
        truncated = False

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
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

            score_text = FONT.render(f"Score: {int(self.score)}", True, TEXT_COLOR)
            self.screen.blit(score_text, (10, 10))

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# ----------------------
# Main execution for testing
# ----------------------
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
