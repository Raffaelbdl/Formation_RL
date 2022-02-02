import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym
import learnrl as rl
from MEMORY import Memory
kl = tf.keras.layers


class REINFORCE(rl.Agent):

    def __init__(self, actor: tf.keras.Model):

        MEMORY_KEYS = ['observation', 'action',
                       'reward', 'done', 'next_observation']
        self.memory = Memory(MEMORY_KEYS=MEMORY_KEYS)

        self.actor = actor
        self.opt = tf.keras.optimizers.Adam(1e-4)
        self.gamma = 0.99

    def act(self, observation, **kwargs):
        # Il faut créer un batch d'observation de taille 1
        # size (1, observation_space)
        #
        # ===

        # pi(a|s)
        # vous pourriez avoir besoin de tfp.distributions.Categorical
        #
        # ===

        # a ~ pi(a|s)
        # sample avec la méthode sample
        #
        # ===

        return action

    def learn(self, **kwargs):

        observations, actions, rewards, dones, next_observations = self.memory.sample(
            method='all')
        metrics = {}

        # On n'apprend que si la dernière transition est une fin d'épisode
        if dones[-1]:
            ep_length = len(self.memory)

            # On crée un tensor contenant le numéro de la transition et l'indice de l'action
            # (ep_len, 2)
            #
            # ===

            # On crée le tensor des gains en commençant par la fin
            # G_{t-1} = gamma * Rt + G_{t}
            #
            # ===

            # On calcule la loss dans un GradientTape pour obtenir les gradients
            with tf.GradientTape() as tape:
                # loss = - mean(G * logprobs)
                #
                # ===

            grads = tape.gradient(loss, self.actor.trainable_weights)
            self.opt.apply_gradients(zip(grads, self.actor.trainable_weights))
            metrics['actor_loss'] = tf.reduce_mean(loss).numpy()

            # On vide la mémoire à la fin de l'apprentissage
            self.memory.empty()

        return metrics

    def remember(self, observation, action, reward, done, next_observation, info={}, **kwargs):
        self.memory.remember((observation, action, reward,
                              done, next_observation))


if __name__ == "__main__":

    env = gym.make("CartPole-v0")

    actor = tf.keras.models.Sequential([
        kl.Dense(16, activation='tanh'),
        kl.Dense(16, activation='tanh'),
        kl.Dense(env.action_space.n, activation='softmax')
    ])
    agent = REINFORCE(actor=actor)

    episodes = 10000

    metrics = [('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
               ('actor_loss')]

    pg = rl.Playground(env, [agent])
    pg.fit(episodes, verbose=2, metrics=metrics)
    pg.run(5)
