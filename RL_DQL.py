# DEEP Q-LEARNING REINFORCEMENT LEARNING ALGORITHM
# ================================================

import random
from game import SnakeGame
from utils import *
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

MEM_LIMIT = 50000
BATCH_SIZE = 1000
SCORE_TO_SAVE_THRESHOLD = 20


class DeepQNet(nn.Module):

    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.af1 = nn.ReLU()
        self.linear2 = nn.Linear(mid_dim, out_dim)
        #self.af2 = nn.ReLU()

    def forward(self, x):
        x = self.af1(self.linear1(x))
        #x = self.af2(self.linear2(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        self.load_state_dict(torch.load(file_name))


class DeepQTrainer:
    def __init__(self, model, lr=0.001, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, state_old, action, state_new, reward, game_over):
        state_old = torch.tensor(state_old, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        state_new = torch.tensor(state_new, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        if len(state_old.shape) == 1:
            state_old = torch.unsqueeze(state_old, 0)
            action = torch.unsqueeze(action, 0)
            state_new = torch.unsqueeze(state_new, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        pred = self.model(state_old)
        target = pred.clone()
        for idx in range(len(game_over)):
            q_new = reward[idx]
            if not game_over[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(state_new[idx]))
            target[idx][torch.argmax(action[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.loss_fn(target, pred)
        loss.backward()

        self.optimizer.step()


class QLearningAgent:

    def __init__(self, game, load_model=False):
        self.model = DeepQNet(12, 128, 4)
        self.trainer = DeepQTrainer(self.model)
        self.memory = deque(maxlen=MEM_LIMIT)
        self.game = game

    # The new_head must be different from the true head by at most 1 unit
    def _potential_collision(self, new_head):
        if new_head.x >= self.game.w or new_head.x < 0 or new_head.y >= self.game.h or new_head.y < 0:
            return True
        elif new_head in self.game.snake[:-1]:
            return True
        else:
            return False

    def get_state(self):
        # input: 
        #    [
        #    danger near:[STRAIGHT, RIGHT, BACK, LEFT], 
        #    direction:[LEFT, RIGHT, UP, DOWN], 
        #    food direction:[LEFT, RIGHT, UP, DOWN]
        #    ]
        head = self.game.snake[0]
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)
        
        dir_l = self.game.direction == Direction.LEFT
        dir_r = self.game.direction == Direction.RIGHT
        dir_u = self.game.direction == Direction.UP
        dir_d = self.game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self._potential_collision(point_r)) or 
            (dir_l and self._potential_collision(point_l)) or 
            (dir_u and self._potential_collision(point_u)) or 
            (dir_d and self._potential_collision(point_d)),

            # Danger right
            (dir_u and self._potential_collision(point_r)) or 
            (dir_d and self._potential_collision(point_l)) or 
            (dir_l and self._potential_collision(point_u)) or 
            (dir_r and self._potential_collision(point_d)),

            # Danger back
            True,

            # Danger left
            (dir_d and self._potential_collision(point_r)) or 
            (dir_u and self._potential_collision(point_l)) or 
            (dir_r and self._potential_collision(point_u)) or 
            (dir_l and self._potential_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.game.food.x < self.game.head.x,  # food left
            self.game.food.x > self.game.head.x,  # food right
            self.game.food.y < self.game.head.y,  # food up
            self.game.food.y > self.game.head.y  # food down
            ]
        return np.array(state, dtype=int)

    def get_action(self, state, eps):
        r = random.uniform(0, 1)
        action = np.zeros(4)
        if r > eps:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            action[move] = 1
        else: # random move
            action[random.randint(0, 3)] = 1
        return action

    def train_one_step(self, state_old, action, state_new, reward, game_over):
        self.trainer.train_step(state_old, action, state_new, reward, game_over)

    def train_long_memory(self):
        sample = random.sample(self.memory, k=min(len(self.memory), BATCH_SIZE))
        states_old, actions, states_new, rewards, games_over = zip(*sample)
        states_old = np.array(states_old)
        actions = np.array(actions)
        states_new = np.array(states_new)
        rewards = np.array(rewards)
        self.trainer.train_step(states_old, actions, states_new, rewards, games_over)
    
    def remember(self, state_old, action, state_new, reward, game_over):
        self.memory.append((state_old, action, state_new, reward, game_over))


def train(load_model=False):
    game = SnakeGame()
    agent = QLearningAgent(game)
    nb_games = 0
    if load_model:
        agent.model.load()

    while True:
        game.update_ui()
        if nb_games > 500 or load_model:
            eps = 0
        else:
            eps = 1.0 / np.sqrt(nb_games+1)

        state_old = agent.get_state()
        action = agent.get_action(state_old, eps)
        reward, game_over, score = game.play_step(action)
        state_new = agent.get_state()

        agent.train_one_step(state_old, action, state_new, reward, game_over)
        agent.remember(state_old, action, state_new, reward, game_over)

        if game_over:
            nb_games += 1
            game.reinit()
            agent.train_long_memory()
            if score == game.high_score:
                if score > SCORE_TO_SAVE_THRESHOLD:
                    agent.model.save()

if __name__ == '__main__':
    train(load_model=False)