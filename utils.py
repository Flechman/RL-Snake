from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    DOWN = 4
    LEFT = 2
    UP = 3
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLACK = (0,0,0)
PRIMARY_COLOR_1 = WHITE
PRIMARY_COLOR_2 = BLACK

# game constants
BLOCK_SIZE = 20 # in pixels
W = 32
H = 24
SPEED = 200
SNAKE_START_LEN = 3
NOT_EAT_MOVE_LIMIT = 2000

# simulation/learning constants
REWARD_EAT = 10
REWARD_COLLISION = -10
REWARD_OTHER = 0