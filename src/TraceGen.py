__author__ = 'umanoidTyphoon'

from src.ObservedECCSM import ObservedECCScalarMultiplication
from src.ObservedECCSM import ObservedRandomizedECCScalarMultiplication
from src.KeyGen import KeyGen

import random


class TraceGen():
    def __init__(self, trace_length=1):
        self.trace_length = trace_length
        self.trace_list = []

    def generate(self):
        counter = 0

        key_generator = KeyGen(5)
        key = key_generator.run()

        while counter < self.trace_length:
            random_point = random.SystemRandom().randint(0, 100)
            print "P: ", random_point

            eccsm  = ObservedECCScalarMultiplication(key, random_point)
            reccsm = ObservedRandomizedECCScalarMultiplication(key, random_point)

            print eccsm
            print reccsm

            print "ECCSM result: %d"  % eccsm.run()
            print "RECCSM result: %d" % reccsm.run()

            print "ECCSM observations: ", eccsm.get_obs()
            print "RECCSM observation: ", reccsm.get_obs()

            self.trace_list.append(reccsm.get_obs())

            counter += 1
