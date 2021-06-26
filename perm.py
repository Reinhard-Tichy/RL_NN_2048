"""Prioritized experience replay memory."""

import numpy as np
import random
import megengine.functional as F

class SumTree(object):

    def __init__(self, capacity):
        self.capacity = capacity
        # number of nodes in the tree
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0 # total cnt of current valid data
        self.write = 0 # current new idx for data to be added

    # leaf node retrieval function
    # idx - index of the top-parent node as the first arguments
    # value - random sampled value
    def retrieve(self, idx, value):
        # left/right - left/right idx of elements following after idx
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self.retrieve(left, value)
        else:
            return self.retrieve(right, value - self.tree[left])

    # update priority
    def update(self, idx, new_value):
        change = new_value - self.tree[idx]
        self.tree[idx] = new_value
        self.propagate_changes(idx, change)

    def propagate_changes(self, idx, change):
 
        # calculate parent id
        parent = (idx - 1) // 2
        self.tree[parent] += change

        # if we not in root propogate changes
        if parent != 0:
            self.propagate_changes(parent, change)

    def add(self, value, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, value)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # get priority and sample
    def get(self, value):
        idx = self.retrieve(0, value)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.n_entries


class perm:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def get_priority(self, error):
        return (error + self.e) ** self.a

    def append(self, data, error):
        value = self.get_priority(error)
        #print(type(value), value)
        self.tree.add(value, data)

    def sample_batch(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):

            left_border = segment * i
            right_border = segment * (i + 1)

            value = random.uniform(left_border, right_border)
            (idx, priority, data) = self.tree.get(value)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        res = []
        for i in range(len(batch[0])):
            k = F.stack(tuple(item[i] for item in batch), axis=0)
            res.append(k)

        return tuple(res), idxs, is_weight

    def update(self, idx, error):
        priority = self.get_priority(error)
        self.tree.update(idx, priority)

    def size(self):
        return len(self.tree)


# https://github.com/megvii-research/ICCV2019-LearningToPaint/blob/master/baseline/DRL/rpm.py

class rpm(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

    def append(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        res = []
        for i in range(6):
            k = F.stack(tuple(item[i] for item in batch), axis=0)
            res.append(k)
        return res[0], res[1], res[2], res[3], res[4], res[5]