import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import deque

# Rozmiary okna gry (ustawienia preferowane powinny lepiej działać w teście nauki)
WIDTH, HEIGHT = 600, 400
CELL_SIZE = 20

# Kolory (zielony to gracz)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Kierunki ruchu
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Model AI
class SnakeAI(nn.Module):
    def __init__(self):
        super(SnakeAI, self).__init__()
        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Buffer doświadczeń (NAUKA)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

def ai_move(snake, apple, epsilon):
    state = get_state(snake, apple)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    if random.random() < epsilon:
        action = random.randint(0, 3)
    else:
        with torch.no_grad():
            action = ai(state_tensor).argmax().item()
    directions = [RIGHT, LEFT, DOWN, UP]
    return directions[action], action

# Funkcja do obliczania stanu gry dla AI
def get_state(snake, apple):
    head_x, head_y = snake.body[0]
    state = [
        int((head_x + CELL_SIZE, head_y) in snake.body or head_x + CELL_SIZE >= WIDTH),  # Prawo
        int((head_x - CELL_SIZE, head_y) in snake.body or head_x - CELL_SIZE < 0),       # Lewo
        int((head_x, head_y + CELL_SIZE) in snake.body or head_y + CELL_SIZE >= HEIGHT), # Dół
        int((head_x, head_y - CELL_SIZE) in snake.body or head_y - CELL_SIZE < 0),       # Góra
        int(head_x >= WIDTH - CELL_SIZE),  # Blisko prawej ściany
        int(head_x < CELL_SIZE),           # Blisko lewej ściany
        int(head_y >= HEIGHT - CELL_SIZE), # Blisko dolnej ściany
        int(head_y < CELL_SIZE)            # Blisko górnej ściany
    ]

    if len(snake.body) > 1:
        neck_x, neck_y = snake.body[1]
    else:
        neck_x, neck_y = head_x - snake.direction[0] * CELL_SIZE, head_y - snake.direction[1] * CELL_SIZE

    state += [
        int(neck_x > head_x),  # Głowa porusza się w lewo
        int(neck_x < head_x),  # Głowa porusza się w prawo
        int(neck_y > head_y),  # Głowa porusza się w górę
        int(neck_y < head_y)   # Głowa porusza się w dół
    ]

    state += [
        int(apple[0] > head_x),     # Jabłko po prawej
        int(apple[0] < head_x),     # Jabłko po lewej
        int(apple[1] > head_y),     # Jabłko poniżej
        int(apple[1] < head_y)      # Jabłko powyżej
    ]

    return np.array(state, dtype=int)

# Klasa Węża
class Snake:
    def __init__(self, x, y):
        self.body = [(x, y)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def move(self):
        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx * CELL_SIZE, head_y + dy * CELL_SIZE)
        self.body = [new_head] + self.body[:-1]

    def grow(self):
        self.body.append(self.body[-1])

    def check_collision(self):
        head = self.body[0]
        return (
            head[0] < 0 or head[0] >= WIDTH or
            head[1] < 0 or head[1] >= HEIGHT or
            head in self.body[1:]
        )

# Funkcja do resetowania węża AI
def reset_ai(snake):
    snake.body = [(random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE,
                   random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE)]
    snake.direction = random.choice([UP, DOWN, LEFT, RIGHT])

# Funkcja do rysowania liczby żyć
def draw_lives(lives):
    font = pygame.font.SysFont(None, 36)
    text = font.render(f'AI Lives: {lives}', True, WHITE)
    screen.blit(text, (10, 10))

# Funkcja do rysowania licznika czasu
def draw_timer(seconds):
    font = pygame.font.SysFont(None, 36)
    text = font.render(f'Time: {int(seconds)}s', True, WHITE)
    screen.blit(text, (WIDTH - 150, 10))

# Funkcja wyświetlająca komunikat końcowy
def display_end_message(message):
    screen.fill(BLACK)
    font = pygame.font.SysFont(None, 48)
    text = font.render(message, True, WHITE)
    screen.blit(text, (WIDTH // 4, HEIGHT // 3))
    font_small = pygame.font.SysFont(None, 36)
    restart_text = font_small.render("Naciśnij R, aby rozpocząć od nowa lub Q, aby zakończyć.", True, WHITE)
    screen.blit(restart_text, (WIDTH // 6, HEIGHT // 2))
    pygame.display.flip()

    # Wstrzymaj grę, dopóki użytkownik nie zdecyduje
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "restart"
                elif event.key == pygame.K_q:
                    pygame.quit()
                    return "quit"

# Funkcja główna gry
def main():
    global ai, target_ai, optimizer, criterion, epsilon, gamma, buffer, batch_size, target_update
    ai = SnakeAI()
    target_ai = SnakeAI()
    target_ai.load_state_dict(ai.state_dict())
    optimizer = optim.Adam(ai.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epsilon = 1.0  # Początkowa wartość eksploracji
    epsilon_min = 0.1  # Minimalna wartość eksploracji
    epsilon_decay = 0.995  # Tempo zmniejszania eksploracji
    gamma = 0.9  # Współczynnik dyskontowy
    buffer = ReplayBuffer(capacity=10000)  # Bufor doświadczeń
    batch_size = 64  # Rozmiar paczki do nauki
    target_update = 100  # Co ile kroków aktualizować sieć docelową

    running = True
    steps_done = 0
    while running:
        snake1 = Snake(WIDTH // 2, HEIGHT // 2)
        snake2 = Snake(WIDTH // 4, HEIGHT // 4)
        apple = (random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE,
                 random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE)
        ai_lives = 5
        game_start_time = time.time()
        game_duration = 300  # Czas gry w sekundach (5 minut)

        while True:
            if not running:
                break  # Wyjście z pętli gry, jeśli okno zostało zamknięte

            screen.fill(BLACK)
            elapsed_time = time.time() - game_start_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        return

            if elapsed_time >= game_duration:
                # Koniec gry - ogłoszenie zwycięzcy
                if len(snake1.body) > len(snake2.body):
                    result = display_end_message("Gratulacje! Gracz wygrał!")
                elif len(snake2.body) > len(snake1.body):
                    result = display_end_message("AI wygrało!")
                else:
                    result = display_end_message("Remis!")

                if result == "restart":
                    break
                elif result == "quit":
                    pygame.quit()
                    return

            if ai_lives > 0:
                # Ruch sterowanego węża
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP] and snake1.direction != DOWN:
                    snake1.direction = UP
                elif keys[pygame.K_DOWN] and snake1.direction != UP:
                    snake1.direction = DOWN
                elif keys[pygame.K_LEFT] and snake1.direction != RIGHT:
                    snake1.direction = LEFT
                elif keys[pygame.K_RIGHT] and snake1.direction != LEFT:
                    snake1.direction = RIGHT

                snake1.move()

                # Ruch węża AI
                direction, action = ai_move(snake2, apple, epsilon)
                snake2.direction = direction
                snake2.move()

                if snake1.check_collision():
                    display_end_message("Gratulacje! AI wygrało!")
                    break

                if snake2.check_collision():
                    ai_lives -= 1
                    reset_ai(snake2)
                    if ai_lives <= 0:
                        display_end_message("AI przegrało!")
                        break

                # Sprawdzenie, czy jabłko zostało zjedzone
                if snake1.body[0] == apple:
                    snake1.grow()
                    apple = (random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE,
                             random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE)

                if snake2.body[0] == apple:
                    snake2.grow()
                    apple = (random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE,
                             random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE)

                # Update Q-tabeli
                next_state = get_state(snake2, apple)
                reward = 10 if snake2.body[0] == apple else -10 if snake2.check_collision() else 0
                buffer.push((get_state(snake2, apple), action, reward, next_state))

                if len(buffer) >= batch_size:
                    experiences = buffer.sample(batch_size)
                    batch_state, batch_action, batch_reward, batch_next_state = zip(*experiences)

                    batch_state_tensor = torch.FloatTensor(batch_state)
                    batch_action_tensor = torch.LongTensor(batch_action)
                    batch_reward_tensor = torch.FloatTensor(batch_reward)
                    batch_next_state_tensor = torch.FloatTensor(batch_next_state)

                    # Obliczanie Q wartości
                    current_q = ai(batch_state_tensor).gather(1, batch_action_tensor.unsqueeze(1)).squeeze(1)
                    next_q = target_ai(batch_next_state_tensor).max(1)[0]
                    target_q = batch_reward_tensor + gamma * next_q

                    loss = criterion(current_q, target_q)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Aktualizacja sieci docelowej
                    if steps_done % target_update == 0:
                        target_ai.load_state_dict(ai.state_dict())

                    steps_done += 1

                # Zmniejszenie wartości epsilon
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                # Rysowanie węży, jabłka i licznika czasu
                pygame.draw.rect(screen, RED, (*apple, CELL_SIZE, CELL_SIZE))
                for x, y in snake1.body:
                    pygame.draw.rect(screen, GREEN, (x, y, CELL_SIZE, CELL_SIZE))
                for x, y in snake2.body:
                    pygame.draw.rect(screen, WHITE, (x, y, CELL_SIZE, CELL_SIZE))

                draw_lives(ai_lives)
                draw_timer(game_duration - elapsed_time)

                pygame.display.flip()
                clock.tick(10)

if __name__ == "__main__":
    main()
