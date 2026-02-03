import gymnasium as gym
import numpy as np
import random
import time
from tqdm import tqdm  # Progress bar

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
ENV_NAME = "Taxi-v3"
TRAIN_EPISODES = 2000
MAX_STEPS = 100
LEARNING_RATE = 0.7
DISCOUNT_RATE = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.005
EPSILON_MIN = 0.01

def watch_agent(qtable=None, delay=0.1):
    """
    Runs one episode visually. 
    If qtable is None, the agent acts randomly (Untrained).
    If qtable is provided, the agent acts smartly (Trained).
    """
    # Initialize with human rendering to see the taxi
    env = gym.make(ENV_NAME, render_mode="human")
    state, info = env.reset()
    done = False
    total_reward = 0
    
    print("\n🎬 Simulation Started...")
    
    for step in range(MAX_STEPS):
        if qtable is None:
            # 🤪 UNTRAINED: Pick a random action
            action = env.action_space.sample()
        else:
            # 🧠 TRAINED: Pick the best known action (Exploit)
            action = np.argmax(qtable[state, :])
            
        new_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = new_state
        
        time.sleep(delay) # Slow down so we can see what's happening
        
        if done:
            break
            
    print(f"🏁 Episode finished. Total Score: {total_reward}\n")
    env.close()

def train_agent():
    """
    Trains the Q-Learning agent without rendering (for speed).
    """
    env = gym.make(ENV_NAME, render_mode=None)
    
    # Initialize Q-Table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))
    
    epsilon = EPSILON_START

    print(f"🔄 Training for {TRAIN_EPISODES} episodes...")
    for _ in tqdm(range(TRAIN_EPISODES)):
        state, info = env.reset()
        done = False
        
        for _ in range(MAX_STEPS):
            # Explore vs Exploit
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state, :])

            new_state, reward, terminated, truncated, info = env.step(action)
            
            # Q-Learning Update
            current_q = qtable[state, action]
            max_future_q = np.max(qtable[new_state, :])
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q - current_q)
            qtable[state, action] = new_q
            
            state = new_state
            if terminated or truncated:
                break
        
        epsilon = max(EPSILON_MIN, epsilon - EPSILON_DECAY)
        
    env.close()
    return qtable

# ==========================================
# 🚀 MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    print(f"🚕 Welcome to the {ENV_NAME} Q-Learning Demo!")
    
    # --- STEP 1: SEE UNTRAINED AGENT ---
    input("\n❌ Press [Enter] to watch the UNTRAINED (random) agent fail...")
    watch_agent(qtable=None, delay=0.2)
    
    # --- STEP 2: TRAIN ---
    input("💪 Press [Enter] to TRAIN the agent...")
    trained_qtable = train_agent()
    print("✅ Training Complete!")

    # --- STEP 3: SEE TRAINED AGENT ---
    while True:
        input("🏆 Press [Enter] to watch the TRAINED agent perform...")
        watch_agent(qtable=trained_qtable, delay=0.3)
        print("Run again? (Ctrl+C to exit)")