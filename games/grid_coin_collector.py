import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random

class GridWorldEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(GridWorldEnv, self).__init__()
        
        # Налаштування гри
        self.GRID_WIDTH = 15
        self.GRID_HEIGHT = 10
        self.TILE_SIZE = 40
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Дії: 0=Вліво, 1=Вправо, 2=Вгору, 3=Вниз
        self.action_space = spaces.Discrete(4)
        
        # Стан: координати гравця (x, y) та монети (x, y)
        self.observation_space = spaces.Box(low=0, high=15, shape=(4,), dtype=int)

        # Карта (1 = стіна, 0 = прохід)
        self.LEVEL_MAP = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1,0,0,0,0,0,0,0,1],
            [1,0,1,1,0,0,1,0,1,1,1,0,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,1,0,1,0,1],
            [1,1,0,1,1,1,0,1,1,0,1,0,0,0,1],
            [1,0,0,0,0,0,0,0,1,0,0,0,1,0,1],
            [1,0,1,1,1,0,1,0,1,1,1,0,1,0,1],
            [1,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
            [1,0,1,0,0,0,1,1,1,0,1,1,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Гравець стартує в точці (1, 1)
        self.player_pos = np.array([1, 1])
        
        # Монета (FIX: ставимо її завжди в одне місце, щоб агент швидше вчився)
        self.coin_pos = np.array([13, 8]) 
        
        self.moves = 0
        
        observation = np.concatenate((self.player_pos, self.coin_pos))
        return observation, {}

    def step(self, action):
        # Логіка руху (0=Вліво, 1=Вправо, 2=Вгору, 3=Вниз)
        direction = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([0, 1])
        }
        
        current_pos = self.player_pos
        delta = direction.get(action, np.array([0, 0]))
        new_pos = current_pos + delta
        
        # Перевірка стін
        if self.LEVEL_MAP[new_pos[1]][new_pos[0]] == 1:
            new_pos = current_pos

        self.player_pos = new_pos
        self.moves += 1
        
        # Винагорода
        reward = -1  # Штраф за кожен крок (стимул бігти швидше)
        terminated = False
        
        # Чи взяв монету?
        if np.array_equal(self.player_pos, self.coin_pos):
            reward = 100 # Велика нагорода
            terminated = True 
            
        # Обмеження ходів (якщо заблукав)
        if self.moves >= 200:
            terminated = True
            
        observation = np.concatenate((self.player_pos, self.coin_pos))
        return observation, reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.GRID_WIDTH * self.TILE_SIZE, self.GRID_HEIGHT * self.TILE_SIZE))
            pygame.display.set_caption("AI Learning Grid World")
            self.clock = pygame.time.Clock()

        self.window.fill((240, 240, 240)) 

        # Малюємо карту
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(x*self.TILE_SIZE, y*self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                if self.LEVEL_MAP[y][x] == 1:
                    pygame.draw.rect(self.window, (100, 100, 100), rect) 
                else:
                    pygame.draw.rect(self.window, (0, 0, 0), rect, 1)

        # Малюємо монету
        pygame.draw.circle(
            self.window, (255, 215, 0), 
            (self.coin_pos[0]*self.TILE_SIZE + self.TILE_SIZE//2, self.coin_pos[1]*self.TILE_SIZE + self.TILE_SIZE//2),
            self.TILE_SIZE//4
        )

        # Малюємо агента
        pygame.draw.rect(
            self.window, (50, 100, 255), 
            (self.player_pos[0]*self.TILE_SIZE, self.player_pos[1]*self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        )

        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.window is not None:
            pygame.quit()