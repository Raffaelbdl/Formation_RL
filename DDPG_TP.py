from re import M
import numpy as np
import tensorflow as tf
import gym
from MEMORY import Memory
kl = tf.keras.layers
ki = tf.keras.initializers


class DDPG():

    def __init__(
        self, 
        env: gym.Env,
        actor: tf.keras.Model,
        target_actor: tf.keras.Model,
        critic: tf.keras.Model,
        target_critic: tf.keras.Model
    ):

        MEMORY_KEYS = ['S', 'A', 'R', 'Done', 'S_next']
        self.memory = Memory(MEMORY_KEYS=MEMORY_KEYS, max_memory_len=40960)

        self.actor = actor
        self.target_actor = target_actor
        self.actor_opt = tf.keras.optimizers.Adam(1e-3)

        self.critic = critic
        self.target_critic = target_critic
        self.critic_opt = tf.keras.optimizers.Adam(1e-3)

        self.exploration = 0.05
        self.gamma = 0.99
        self.polyak = 0.995
        self.sample_size = 4096

        self.action_high = env.action_space.high
        self.action_low = env.action_space.low
        self.num_action = env.action_space.shape[0]

    def act(self, s, greedy=False):

        A = None
        return tf.clip_by_value(A, self.action_low, self.action_high).numpy()[0]

    def compute_targets(self, R, Done, S_next):

        Tar = None
        return Tar

    def improve(self):
        
        metrics = {}

        metrics.update(self.update_value(S, A, R, Done, S_next))
        metrics.update(self.update_actor(S))

        self.update_target(
            current=self.critic,
            target=self.target_critic
        )
        self.update_target(
            current=self.actor, 
            target=self.target_actor
        )
        return metrics

    def update_value(self, S, A, R, Done, S_next):

        loss, Q  = None, None
        return {
            'value_loss': loss.numpy(),
            'value': tf.reduce_mean(Q).numpy()
        }

    def update_actor(self, S):

        loss = None
        return {'actor_loss': loss.numpy()}

    def update_network(self, network: tf.keras.Model, loss, tape, opt: tf.keras.optimizers.Optimizer):
        grads = tape.gradient(loss, network.trainable_weights)
        opt.apply_gradients(zip(grads, network.trainable_weights))

    def update_target(self, current: tf.keras.Model, target: tf.keras.Model):
        new_weights = self.polyak * \
            np.array(target.get_weights()) + (1-self.polyak) * \
            np.array(current.get_weights())
        target.set_weights(new_weights)

    def add(self, s, a, r, done, s_next):
        r = float(r)
        self.memory.add((s, a, r, done, s_next))


if __name__ == "__main__":

    # env = gym.make("Pendulum-v1")
    env = gym.make("LunarLanderContinuous-v2")
    input_shape = (env.observation_space.shape[0] + env.action_space.shape[0],)
    num_action = env.action_space.shape[0]
    obs_shape = env.observation_space.shape
    print(num_action)
    print(obs_shape)

    actor = tf.keras.models.Sequential([
        ...,
        kl.Dense(...)
    ])
    target_actor = tf.keras.models.clone_model(actor)

    critic = tf.keras.models.Sequential([
        ...,
        kl.Dense(...)
    ])
    target_critic = tf.keras.models.clone_model(critic)

    
    agent = DDPG(
        env=env, 
        actor=actor, 
        target_actor=target_actor,
        critic=critic, 
        target_critic=target_critic
    )

    episodes = 1000

    for ep in range(episodes):

        s = env.reset()
        done = False
        ep_r = 0
        while not done:
            a = agent.act(s) * env.action_space.high
            s_next, r, done, info = env.step(a)

            ep_r += r

            agent.add(s, a, r, done, s_next)

            metrics = agent.improve()

            s = s_next
            env.render()

        value, a_loss, c_loss = metrics["value"], metrics["actor_loss"], metrics["value_loss"]
        print(
            f"Ep {ep+1}/{episodes} | Ep-rwd {ep_r} | V {value} | A Loss {a_loss:.2f} | C Loss {c_loss:.2f}"
        )