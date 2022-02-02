import tensorflow as tf


class Memory():

    def __init__(self, MEMORY_KEYS: list, max_memory_len: int = None):
        self.MEMORY_KEYS = MEMORY_KEYS
        self.trajectory = {}
        self.max_memory_len = max_memory_len

    def remember(self, transition: tuple):
        for val, key in zip(transition, self.MEMORY_KEYS):
            batched_val = tf.expand_dims(val, axis=0)
            try:
                self.trajectory[key] = tf.concat(
                    (self.trajectory[key], batched_val), axis=0)
            except KeyError:
                self.trajectory[key] = batched_val

            if self.max_memory_len != None:
                self.trajectory[key] = self.trajectory[key][-self.max_memory_len:]

        self.memory_len = len(self.trajectory[self.MEMORY_KEYS[0]])

    def sample(self, method='last', sample_size=None):
        if method == 'all':
            trajectory = [self.trajectory[key] for key in self.MEMORY_KEYS]

        elif method == 'last':
            trajectory = [self.trajectory[key][-sample_size:]
                          for key in self.MEMORY_KEYS]

        elif method == 'random':
            indexes = tf.random.shuffle(tf.range(self.memory_len))[
                :sample_size]
            trajectory = [tf.gather(self.trajectory[key], indexes)
                          for key in self.MEMORY_KEYS]
        else:
            raise NotImplementedError('Not implemented sample')

        return trajectory

    def __len__(self):
        return self.memory_len

    def empty(self):
        self.trajectory = {}
