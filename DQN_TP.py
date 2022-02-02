import tensorflow as tf
import gym
import learnrl as rl
from MEMORY import Memory
kl = tf.keras.layers


class DQN(rl.Agent):

    def __init__(self, action_value: tf.keras.Model):

        MEMORY_KEYS = ['observation', 'action',
                       'reward', 'done', 'next_observation']
        self.memory = Memory(MEMORY_KEYS=MEMORY_KEYS, max_memory_len=40960)

        self.action_value = action_value
        self.opt = tf.keras.optimizers.Adam(1e-4)

        self.exploration = 0.1
        self.gamma = 0.99
        self.sample_size = 4096

    def act(self, observation, greedy=False):
        # Il faut créer un batch d'observation de taille 1
        # size (1, observation_space)
        #
        # ===

        # Evaluons Q(s)
        # size (1, action_space)
        #
        # ===

        # Prenons une action
        # Si Greedy :
        if greedy:
            #
            pass
            # ===
        # Sinon Epsilon greedy :
        else:
            #
            pass
            # ===

        return action

    def evaluate(self, rewards, dones, next_observations):
        futur_rewards = rewards

        # On regarde là où les épisodes ne sont pas finis (1-d = 1)
        ndones = tf.logical_not(dones)
        if tf.reduce_any(ndones):
            # next_value = max(Q(s_next))

            # y = r + gamma*max(Q)
            # vous pourriez avoir besoin de tf.tensor_scatter_nd_add

            pass
            # ===

        return futur_rewards

    def learn(self):
        observations, actions, rewards, dones, next_observations = self.memory.sample(
            method='random', sample_size=self.sample_size)

        # On calcule gamma * Q(a', s') "expected_futur_rewards"
        # on pourra réutiliser la méthode evaluate
        #
        # ===

        # on calcule la loss
        with tf.GradientTape() as tape:
            # calcul de Q(a,s)
            # on récupère les valeurs de Q associées aux actions
            # vous pourriez avoir besoin de tf.gather_nd
            #
            # ===

            loss = tf.keras.losses.mse(expected_futur_rewards, Q_action)

        grads = tape.gradient(loss, self.action_value.trainable_weights)
        self.opt.apply_gradients(
            zip(grads, self.action_value.trainable_weights))

        metrics = {
            'value': tf.reduce_mean(Q_action).numpy(),
            'loss': loss.numpy()
        }

        return metrics

    def remember(self, observation, action, reward, done, next_observation, info={}, **param):
        self.memory.remember((observation, action, reward,
                              done, next_observation))


if __name__ == "__main__":

    env = gym.make("CartPole-v0")

    action_value = tf.keras.models.Sequential([
        kl.Dense(16, activation='tanh'),
        kl.Dense(16, activation='tanh'),
        kl.Dense(env.action_space.n, activation='linear')
    ])

    agent = DQN(action_value=action_value)

    episodes = 1000

    metrics = [('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
               ('value'), ('loss')]

    pg = rl.Playground(env, [agent])
    pg.fit(episodes, verbose=2, metrics=metrics, render=True)
    pg.run(5)
