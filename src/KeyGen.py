__author__ = 'umanoidTyphoon'

from abc import ABCMeta, abstractmethod
import random


class Algorithm(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, key, ecc_p):
        pass


class KeyGen(Algorithm):
    def __init__(self, key_length=1):
        self.N = key_length

    def run(self):
        random_string = ""
        for _ in xrange(self.N):
            random_bit = str(random.SystemRandom().choice([0, 1]))
            random_string += random_bit
        return random_string