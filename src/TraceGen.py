__author__ = 'umanoidTyphoon'

from src.ObservedECCSM import ObservedECCScalarMultiplication
from src.ObservedECCSM import ObservedRandomizedECCScalarMultiplication

import random
import sys


class TraceGen():
    def __init__(self, key, traces_to_be_generated):
        self.key = key
        self.traces_to_be_generated = traces_to_be_generated
        self.trace_list = []

    def generate(self):
        counter = 0
        random_point = random.SystemRandom().randint(0, sys.maxint)

        while counter < self.traces_to_be_generated:
            print "P: ", random_point

            # eccsm  = ObservedECCScalarMultiplication(self.key, random_point)
            reccsm = ObservedRandomizedECCScalarMultiplication(self.key, random_point)

            # print eccsm
            print reccsm

            # print "ECCSM result: %d"  % eccsm.run()
            print "RECCSM result: %d" % reccsm.run()

            # print "ECCSM observations: ", eccsm.get_obs()
            print "RECCSM observation: ", reccsm.get_obs()

            # self.trace_list.append(eccsm.get_obs())
            self.trace_list.append(reccsm.get_obs())

            counter += 1

        observations = ""
        formatted_trace_list = []

        for observation_list in self.trace_list:
            for observation in observation_list:
                observations += observation + " "
            formatted_trace_list.append(observations)
            observations = ""

        return formatted_trace_list