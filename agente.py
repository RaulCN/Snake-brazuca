import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_00000
BATCH_SIZE = 10000
LR = 0.0001

class Agent:

    def __init__(self):
        self.n_games = 0 # número de jogos jogados
        self.epsilon = 1 # quanto mais perto de 0 mais aletoriedade na exploração
        self.gamma = 0.9 # fator de desconto
        self.memory = deque(maxlen=MAX_MEMORY) # # popleft() - remove e retorna o elemento mais antigo da esquerda na deque
        self.model = Linear_QNet(11, 512, 3)
        self.model.load() # carregar o modelo salvo
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Perigo em linha reta - verifica se há perigo em linha reta em três direções possíveis: direita, esquerda e para cima
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Perigo à direita - verifica se há perigo na direção à direita em três situações diferentes, dependendo da direção atual da cobra
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Perigo à esquerda - verifica se há perigo na direção à esquerda em três situações diferentes, dependendo da direção atual da cobra
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Direção de movimento - verifica se a cobra está se movendo para a esquerda, direita, para cima ou para baixo
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Localização da comida - verifica em que direção a comida está em relação à cabeça da cobra.
            game.food.x < game.head.x,  # comida à esquerda - verifica se a comida está localizada à esquerda da cabeça da cobra
            game.food.x > game.head.x,  # comida à direita - verifica se a comida está localizada à esquerda da cabeça da cobra
            game.food.y < game.head.y,  # comida acima - verifica se a comida está localizada à esquerda da cabeça da cobra
            game.food.y > game.head.y  # comida abaixo - verifica se a comida está localizada à esquerda da cabeça da cobra
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY for alcançado - remove o elemento mais antigo da esquerda da deque quando o limite máximo de memória é atingido

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # # lista de tuplas - uma lista que contém tuplas como elementos
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #Para cada estado, ação, recompensa, próximo estado e estado final na mini amostra:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # movimentos aleatórios: equilíbrio entre exploration / exploitation (difícil tradução para português)
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # obter estado antigo
        state_old = agent.get_state(game)

        # obter movimento
        final_move = agent.get_action(state_old)

        # executar movimento e obter novo estado
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # ttreinar memória curta
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # relembrar
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Treinar memória longa, plotar resultados
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Jogo', agent.n_games, 'Pontuação', score, 'Recorde:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
