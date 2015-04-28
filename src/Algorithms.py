__author__ = 'umanoidTyphoon'

from abc import ABCMeta, abstractmethod


class Algorithm(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, key, ecc_p):
        pass


class ECCScalarMultiplication(Algorithm):

    def run(self, key, ecc_p):
        return


class RandomizedECCScalarMultiplication(Algorithm):

    def run(self, key, ecc_p):
        return