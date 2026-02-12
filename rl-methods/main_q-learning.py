import sys
import os
import numpy as np
import random
import time

# --- 1. Шлях до папки з іграми ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- 2. Імпорт гри ---
try:
    from games.grid_coin_collector import GridWorldEnv
except ImportError:
    print("❌ ПОМИЛКА: Не знайдено файл 'games/grid_coin_collector.py'")
    sys.exit()

# --- 3. Налаштування ---
env = GridWorldEnv(render_mode=None) 
q_table = {}

# Параметри (збільшено кількість епізодів для кращого навчання)
episodes = 3000      # Кількість спроб
alpha = 0.1          # Швидкість навчання
gamma = 0.99         # Важливість майбутнього
epsilon = 1.0        # Дослідження
epsilon_decay = 0.999 # Повільне зменшення випадковості
epsilon_min = 0.05

def get_state_key(state):
    return f"{state[0]}_{state[1]}_{state[2]}_{state[3]}"

print(f"🚀 Починаємо навчання на {episodes} ігор...")

# --- 4. Тренування ---
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_key = get_state_key(state)

        if state_key not in q_table:
            q_table[state_key] = np.zeros(env.action_space.n)

        # Вибір дії
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_key])

        next_state, reward, done, _, _ = env.step(action)
        next_state_key = get_state_key(next_state)

        if next_state_key not in q_table:
            q_table[next_state_key] = np.zeros(env.action_space.n)

        # Оновлення Q-значення
        old_value = q_table[state_key][action]
        next_max = np.max(q_table[next_state_key])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state_key][action] = new_value

        state = next_state
        total_reward += reward

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % 500 == 0:
        print(f"Епізод {episode}: Очки = {total_reward}, Epsilon = {epsilon:.2f}")

print("✅ Тренування завершено!")

# --- 5. Демонстрація ---
print("🎮 Запускаємо результат...")
env.close()
env = GridWorldEnv(render_mode="human")

try:
    while True:
        state, _ = env.reset()
        done = False
        print("🤖 Нова гра...")
        
        while not done:
            env.render()
            state_key = get_state_key(state)
            
            # Тільки розумні ходи
            if state_key in q_table:
                action = np.argmax(q_table[state_key])
            else:
                action = env.action_space.sample() # Якщо раптом незнайомий стан
            
            state, reward, done, _, _ = env.step(action)
            
except KeyboardInterrupt:
    print("Вихід...")
finally:
    env.close()