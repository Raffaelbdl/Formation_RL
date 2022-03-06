import tensorflow as tf
import gym
from MEMORY import Memory
kl = tf.keras.layers


class DQN():

    def __init__(self, action_value: tf.keras.Model):

        MEMORY_KEYS = ['S', 'A', 'R', 'Done', 'S_next']
        self.memory = Memory(MEMORY_KEYS=MEMORY_KEYS, max_memory_len=40960)

        self.action_value = action_value
        self.opt = tf.keras.optimizers.Adam(1e-3)

        self.exploration = 0.15
        self.gamma = 0.99
        self.sample_size = 4096

    def act(self, s, greedy=False):
        # Il faut créer un batch d'observation de taille 1
        S = tf.expand_dims(
            s, axis=0)  # (1, observation_space)

        # Q(s)
        Q = self.action_value(S)  # (1, action_space)

        # Greedy
        if greedy:
            A = tf.argmax(Q, axis=-1, output_type=tf.int32)
        # Epsilon greedy
        else:
            # On tire au sort rd entre 0 et 1
            rd = tf.random.uniform(
                shape=(Q.shape[0],), minval=0, maxval=1, dtype=tf.float32)

            A_greedy = tf.argmax(Q, axis=-1, output_type=tf.int32)
            A_random = tf.random.uniform(
                shape=(Q.shape[0],), minval=0, maxval=Q.shape[-1], dtype=tf.int32)

            # Si rd < epsilon, on prend l'action random, sinon l'action greedy
            A = tf.where(rd < self.exploration,
                         A_random, A_greedy)

        return A.numpy()[0]

    def compute_targets(self, R, Done, S_next):
        Tar = R

        # On regarde là où les épisodes ne sont pas finis (1-d = 1)
        nDone = tf.logical_not(Done)
        if tf.reduce_any(nDone):
            # next_value = max(Q)
            Q_next = tf.reduce_max(self.action_value(
                S_next[nDone]), axis=-1)

            nDone_indexes = tf.where(nDone)
            # y = r + gamma*max(Q)
            Tar = tf.tensor_scatter_nd_add(
                Tar, nDone_indexes, self.gamma * Q_next)

        return Tar

    def improve(self):
        S, A, R, Done, S_next = self.memory.sample(
            method='random', sample_size=self.sample_size)

        # On calcule gamma * Q(a', s')
        Tar = self.compute_targets(
            R, Done, S_next)

        with tf.GradientTape() as tape:
            Q = self.action_value(S)

            action_range = tf.range(len(A))
            action_ind = tf.stack((action_range, A), axis=-1)

            Q_action = tf.gather_nd(Q, action_ind)

            loss = tf.keras.losses.mse(Tar, Q_action)

        grads = tape.gradient(loss, self.action_value.trainable_weights)
        self.opt.apply_gradients(
            zip(grads, self.action_value.trainable_weights))

        metrics = {
            'value': tf.reduce_mean(Q_action).numpy(),
            'loss': loss.numpy()
        }

        return metrics

    def add(self, s, a, r, done, s_next):
        self.memory.add((s, a, r, done, s_next))


if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    num_action = env.action_space.n
    obs_shape = env.observation_space.shape
    print(num_action)
    print(obs_shape)

    action_value = tf.keras.models.Sequential([
        kl.Dense(64, activation='relu'),
        kl.Dense(num_action)
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
