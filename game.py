import pygame
import random
from utils import *

class SnakeGame:
    
    def __init__(self, w=W, h=H, block_size=BLOCK_SIZE, snake_start_len=SNAKE_START_LEN):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.snake_start_len = snake_start_len
        self.high_score = 0
        pygame.init()
        self.font = pygame.font.Font('arial.ttf', 25)
        self.display = pygame.display.set_mode((w*block_size, h*block_size))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reinit()
        
    def reinit(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = []
        for i in range(min(self.snake_start_len, self.w/2)):
            self.snake.append(Point(self.head.x-i, self.head.y)) 
        self.score = 0
        self.movement = 0
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, self.w-1)
        y = random.randint(0, self.h-1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        # action = [straight, right, back, left]
        self._update_direction(action)
        score_before = self.score
        self._eat_or_move()
        score_after = self.score
        game_over = self._is_collision()
        if self.movement > NOT_EAT_MOVE_LIMIT:
            game_over = True

        reward = REWARD_OTHER
        if game_over:
            reward = REWARD_COLLISION
        if score_after - score_before > 0:
            reward = REWARD_EAT

        return reward, game_over, self.score
    
    def _update_direction(self, action):
        # action = [straight, right, back, left]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.end()
                quit()
                 
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if action[1] == 1: # right turn
            self.direction = clock_wise[(idx + 1) % 4]
        if action[2] == 1: # back turn
            self.direction = clock_wise[(idx + 2) % 4]
        if action[3] == 1: # left turn
            self.direction = clock_wise[(idx + 3) % 4]


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
            if self.score > self.high_score:
                self.high_score = self.score
            self.movement = 0
            self._place_food()
        else:
            self.movement += 1
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

        self.clock.tick(SPEED)

    def end(self):
        print('High score: ', self.high_score)
        pygame.quit()
