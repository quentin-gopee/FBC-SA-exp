from collections import defaultdict
import numpy as np
import random


def random_numbers(n_sum, n_numbers):
    '''
    Generates a list of n_numbers random numbers between 1 and num_sum-n_numbers that sum to n_num
    '''
    rand_num = np.sort(random.sample(range(1, n_sum), n_numbers-1))
    num_labeled_per_class = [rand_num[0]] + [rand_num[i] - rand_num[i-1] for i in range(1, len(rand_num))] + [n_sum-rand_num[-1]]
    return num_labeled_per_class

def exp_imbalance_l(N, C, gamma):
    '''
    N: total number of samples
    C: number of classes
    gamma: imbalance ratio

    return: list of number of samples per class with exponential imbalance
    '''
    n1 = N*(1-gamma**(-1/(C-1)))/(1-gamma**(-C/(C-1)))
    n_samples = []

    for i in range(C):
        n_samples.append(int(n1*gamma**(-i/(C-1))))

    # add remaining samples to the first classes
    for i in range(N-sum(n_samples)):
        n_samples[i] += 1

    return n_samples

def exp_imbalance_u(m1, C, gamma):
    '''
    m1: number of samples in the majority class
    C: number of classes
    gamma: imbalance ratio

    return: list of number of samples per class with exponential imbalance
    '''
    n_samples = []

    for i in range(C):
        n_samples.append(int(m1*gamma**(-i/(C-1))))

    return n_samples

def count_classes(items):
    class_count = defaultdict(int)
    for item in items:
        class_count[item.label] += 1
    return class_count