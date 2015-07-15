__author__ = 'umanoidTyphoon'

# from src.ObservedECCSM import ObservedECCScalarMultiplication
from src.ecc import EC
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
        # 'a' and 'b' parameters taken from elliptic curve shown in the slides "ECC.pdf" (page 7)
        # Previously 'large_prime' and 'G_x' were taken from slides "NIST_Recommended elliptic curves for federal
        # government use (page 6). Now, since the computation of sqrt() is very time consuming given the size of q,
        # 'large prime' is taken from "ECC.pdf" (page 7)
        a = 2
        b = 2
        # G_x = 1769255009665454326
        # large_prime = 6277101735386680763835789423207666416083908700390324961279
        large_prime = 17
        # Sometimes, given 'large_prime' and 'G_x', is not possible to find a proper p. The ecc.py module in that case
        # raises an exception. In order to avoid this, G_x is fixed. Possible values for G_x are: 16, 13, 10, 9, 6, and
        # 5. The value chosen is 13.
        #
        # G_x = random.SystemRandom().randint(1, large_prime - 1)
        G_x = 13

        elliptic_curve = EC(a, b, large_prime)
        p, _ = elliptic_curve.at(G_x)

        while counter < self.traces_to_be_generated:
            print "\nIDHMM tester >> TraceGenerator :: Elliptic curve point P -", p

            # eccsm  = ObservedECCScalarMultiplication(self.key, random_point)
            reccsm = ObservedRandomizedECCScalarMultiplication(self.key, elliptic_curve, p)

            # print eccsm
            # print reccsm

            # print "ECCSM result: %d"  % eccsm.run()
            print "IDHMM tester >> TraceGenerator :: Elliptic curve point Q -", reccsm.run()

            # print "ECCSM observations: ", eccsm.get_obs()
            print "IDHMM tester >> TraceGenerator :: RECCSM observation: ", reccsm.get_obs()

            # self.trace_list.append(eccsm.get_obs())
            self.trace_list.append(reccsm.get_obs())

            counter += 1

        observations = ""
        formatted_trace_list = []

        for observation_list in self.trace_list:
            iteration = 0
            last_observation = len(observation_list) - 1
            for observation in observation_list:
                if iteration == last_observation:
                    observations += observation
                else:
                    observations += observation + " "
                iteration += 1
            formatted_trace_list.append(observations)
            observations = ""

        return formatted_trace_list