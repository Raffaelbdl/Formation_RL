import tensorflow as tf
import gym
from MEMORY import Memory
kl = tf.keras.layers


class REINFORCE():

    def __init__(self,
                 actor: tf.keras.Model,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99):

        MEMORY_KEYS = ['S', 'A', 'R', 'Done', 'S_next']
        self.memory = Memory(MEMORY_KEYS=MEMORY_KEYS)

        self.actor = actor
        self.opt = tf.keras.optimizers.Adam(learning_rate)
        self.gamma = gamma

    def act(self, s):

        a = None
        return a

    def improve(self):

        return None

    def add(self, s, a, r, done, s_next):
        self.memory.add((s, a, r, done, s_next))


if __name__ == "__main__":

    env = gym.make("CartPole-v0")

    actor = tf.keras.models.Sequential([
        ...
    ])
    agent = REINFORCE(actor=actor)

    episodes = 10000
    for ep in range(episodes):

        s = env.reset()
        done = False
        ep_r = 0
        while not done:
            a = agent.act(s)
            s_next, r, done, info = env.step(a)
            ep_r += r

            agent.memory.add(s, a, r, done, s_next)

            s = s_next

        metrics = agent.improve()
        agent.memory.clear()
        loss = metrics["loss"]
        print(
            f"Episode {ep+1}/{episodes} | Ep-rwd {int(ep_r)} | Loss {loss:.2f}")
