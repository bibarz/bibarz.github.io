import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab


reward_table = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
labels = ['rock', 'paper', 'scissors']


def fixed_policy(move):
    while True:
        yield move


def random_policy(seed=0):
    rng = np.random.RandomState(seed)
    while True:
        yield rng.randint(3)


def avoid_policy(move, seed=0):
    rng = np.random.RandomState(seed)
    options = [i for i in range(3) if i != move]
    while True:
        yield options[rng.randint(2)]


def rps_policy_gradient():
    rival_policy = avoid_policy(0)
    theta1 = np.random.randn(3)
    rewards1 = []
    alpha = 0.5
    for i in range(1000):
        p1 = np.exp(theta1) / np.sum(np.exp(theta1))
        move1 = np.argmax(np.random.rand() <= np.cumsum(p1))
        counter_move = rival_policy.next()
        r1 = reward_table[move1, counter_move]
        print "1", labels[move1], labels[counter_move], r1, theta1, p1
        rewards1.append(r1)
        theta1[move1] += alpha * r1 * (1 - p1[move1])
    pylab.plot(np.cumsum(rewards1), 'r')
    pylab.show()


if __name__ == "__main__":
    rps_policy_gradient()
