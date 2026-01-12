import sys
from pathlib import Path
import traci

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from env.traffic_env import TrafficMDPEnv
from mdp.agent import QLearningAgent

# Number of Simulations run for Learning
EPISODES = 25       
STEPS_PER_EPISODE = 1000
HEADLESS = True         

SUMO_CFG = "sumo/network/grid.sumocfg"

def main():
    # Temporary environment to get TLS IDs
    temp_env = TrafficMDPEnv(SUMO_CFG)
    tls_ids = temp_env.reset_headless() if HEADLESS else temp_env.reset()
    tls_ids = temp_env.tls_ids
    temp_env.close()

    # Initialize Q-learning agent
    agent = QLearningAgent(tls_ids)

    best_reward = float("-inf")

    for episode in range(1, EPISODES + 1):
        env = TrafficMDPEnv(SUMO_CFG)
        state = env.reset_headless() if HEADLESS else env.reset()
        total_reward = 0

        for step in range(1, STEPS_PER_EPISODE + 1):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # print every 50 steps
            if step % 50 == 0:
                print(f"Episode {episode} Step {step} - Current Reward: {reward:.2f}")

            if done:
                print(f"Episode {episode} ended early at step {step}")
                break

        env.close()

        # Track best reward
        if total_reward > best_reward:
            best_reward = total_reward

        print(f"Episode {episode}/{EPISODES} - Total Reward: {total_reward:.2f} - Best so far: {best_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    # Save Q-table
    agent.save("mdp/q_table.pkl")
    print("Training complete. Q-table saved to mdp/q_table.pkl")

if __name__ == "__main__":
    main()
