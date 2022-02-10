import tensorflow as tf
import gym
from MEMORY import Memory
kl = tf.keras.layers


class DQN():

    def __init__(self,
                 action_value: tf.keras.Model,
                 learning_rate: float = 1e-3,
                 exploration: float = 0.15,
                 gamma: float = 0.99,
                 max_memory_len: int = 10000,
                 sample_size: int = 1000):

        MEMORY_KEYS = ['S', 'A', 'R', 'Done', 'S_next']
        self.memory = Memory(MEMORY_KEYS=MEMORY_KEYS,
                             max_memory_len=max_memory_len)

        self.action_value = action_value
        self.opt = tf.keras.optimizers.Adam(learning_rate)

        self.exploration = exploration
        self.gamma = gamma
        self.sample_size = sample_size

    def act(self, s, greedy=False):

        a = None
        return a

    def compute_targets(self, R, Done, S_next):

        Tar = None
        return Tar

    def improve(self):

        return None

    def add(self, s, a, r, done, s_next):
        self.memory.add((s, a, r, done, s_next))


if __name__ == "__main__":

    env = gym.make("Acrobot-v1")

    action_value = tf.keras.models.Sequential([
        ...
    ])

    agent = DQN(action_value=action_value)

    episodes = 250

    for ep in range(episodes):

        s = env.reset()
        done = False
        ep_r = 0
        while not done:
            a = agent.act(s)
            s_next, r, done, info = env.step(a)
            ep_r += r

            agent.memory.add(s, a, r, done, s_next)

            metrics = agent.improve()

            s = s_next
        value, loss = metrics["value"], metrics["loss"]
        print(
            f"Episode {ep+1}/{episodes} | Ep-rwd {int(ep_r)} | Value {value:.2f} | Loss {loss:.2f}")
