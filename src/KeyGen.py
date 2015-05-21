__author__ = 'umanoidTyphoon'

import random


class KeyGen():
    def __init__(self, key_length=1):
        self.N = key_length

    def run(self):
        random_string = ""
        for _ in xrange(self.N):
            random_bit = str(random.SystemRandom().choice([0, 1]))
            random_string += random_bit
        return random_string