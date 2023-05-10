import pygame
import random
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    UP = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLACK = (0,0,0)
PRIMARY_COLOR_1 = WHITE
PRIMARY_COLOR_2 = BLACK

BLOCK_SIZE = 20
W = 32
H = 24
SPEED = 10
SNAKE_START_LEN = 3

class SnakeGame:
    
    def __init__(self, w=W, h=H, block_size=BLOCK_SIZE, snake_start_len=SNAKE_START_LEN):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.snake_start_len = snake_start_len
        pygame.init()
        self.font = pygame.font.Font('arial.ttf', 25)
        self.display = pygame.display.set_mode((w*block_size, h*block_size))
        pygame.display.set_caption('Snake')
        self._reinit()
        
    def _reinit(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = []
        for i in range(min(self.snake_start_len, self.w/2)):
            self.snake.append(Point(self.head.x-i, self.head.y)) 
        self.score = 0
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, self.w-1)
        y = random.randint(0, self.h-1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self):
        self._update_direction()
        self._eat_or_move()
        game_over = self._is_collision()
        return game_over, self.score
    
    def _update_direction(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

    def _eat_or_move(self):
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.UP:
            y -= 1
        self.head = Point(x, y)

        self.snake.insert(0, self.head)
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

    def _is_collision(self):
        # hits boundary
        if self.head.x >= self.w or self.head.x < 0 or self.head.y >= self.h or self.head.y < 0:
            return True
        # hits itself
        elif self.head in self.snake[1:]:
            return True
        else:
            return False
        
    def update_ui(self):
        self.display.fill(PRIMARY_COLOR_1)

        b_s = self.block_size
        
        for pt in self.snake:
            pygame.draw.rect(self.display, PRIMARY_COLOR_2, pygame.Rect(pt.x*b_s, pt.y*b_s, b_s, b_s))
            pygame.draw.rect(self.display, PRIMARY_COLOR_1, pygame.Rect(pt.x*b_s+b_s//4, pt.y*b_s+b_s//4, b_s-b_s//2, b_s-b_s//2))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x*b_s, self.food.y*b_s, b_s, b_s))
        
        text = self.font.render(str(self.score), True, PRIMARY_COLOR_2)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def end(self):
        pygame.quit()
            

if __name__ == '__main__':
    game = SnakeGame()
    clock = pygame.time.Clock()
    
    # game loop
    while True:
        game.update_ui()
        clock.tick(SPEED)
        game_over, score = game.play_step()
        
        if game_over:
            print('Final Score', score)
            game.end()
            break