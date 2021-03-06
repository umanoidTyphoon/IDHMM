__author__ = 'umanoidTyphoon'

from abc import ABCMeta, abstractmethod
import random


class Algorithm(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, key, ecc_p):
        pass


class ObservedECCScalarMultiplication(Algorithm):
    def __init__(self, key, ecc_p):
        self.key = key
        self.ecc_p = ecc_p

        # Observation list:
        # - D: elliptic point doubling
        # - AD: elliptic point addition
        self.obs = []

    def get_obs(self):
        return self.obs

    def run(self):
        ecc_q = self.ecc_p
        p = 0

        for char in self.key:
            bit = int(char)
            if bit == 1:
                p += ecc_q
                # Observation performed
                self.obs.append("AD")
            else:
                self.obs.append("D")
            ecc_q = 2 * ecc_q
        return p

    def __str__(self):
        return 'ECCSM parms: <' + str(self.key) + ', ' + str(self.ecc_p) + '>'


class ObservedRandomizedECCScalarMultiplication(Algorithm):

    def __init__(self, key, ecc_p):
        self.key = key
        self.ecc_p = ecc_p

        # Observation list:
        # - D: elliptic point doubling
        # - AD: elliptic point addition
        self.obs = []

    def get_obs(self):
        return self.obs

    def run(self):
        ecc_q = self.ecc_p
        p = 0

        for char in self.key:
            bit = int(char)
            r = p
            random_bit = random.SystemRandom().choice([0, 1])

            if bit == 0 :
                if random_bit == 1:
                    r += ecc_q
                    self.obs.append("AD")
                else:
                    self.obs.append("D")
            else:
                p += ecc_q
                self.obs.append("AD")
            ecc_q = 2 * ecc_q
        return p

    def __str__(self):
        return 'RECCSM parms: <' + str(self.key) + ', ' + str(self.ecc_p) + '>'