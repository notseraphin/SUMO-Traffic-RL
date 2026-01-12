import random
import pickle

class QLearningAgent:
    def __init__(self, tls_ids, n_phases=4, alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_decay=0.99, epsilon_min=0.05):
        self.tls_ids = tls_ids
        self.n_phases = n_phases
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}  # key: state tuple, value: dict {action tuple: q-value}

    def get_q(self, state, action):
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, state):
        """
        ε-greedy action selection
        """
        if random.random() < self.epsilon:
            return {tls: random.randint(0, self.n_phases-1) for tls in self.tls_ids}
        else:
            q_state = self.q_table.get(state, {})
            if not q_state:
                return {tls: random.randint(0, self.n_phases-1) for tls in self.tls_ids}
            best_action = max(q_state, key=q_state.get)
            return {tls: phase for tls, phase in zip(self.tls_ids, best_action)}

    def update(self, state, action_dict, reward, next_state):
        action = tuple(action_dict[tls] for tls in self.tls_ids)
        next_qs = self.q_table.get(next_state, {})
        max_next_q = max(next_qs.values()) if next_qs else 0.0

        old_q = self.get_q(state, action)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)

        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
