__author__ = 'umanoidTyphoon'

import numpy as np
from numpy import matrix

class IDHMM:
    def __init__(self, key="0", trace_list=[]):
        self.belief = init_belief(key)
        self.transition_models = init_transition_models()
        self.key = key
        self.trace_list = trace_list

    def infer(self):
        multitrace_inference(self.belief, self.key, self.trace_list)


def init_belief(key):
    key_length = len(key)
    belief = np.empty(key_length)
    belief.fill(.5)

    return belief


def init_transition_models():
    models = []
    # Key bit is 0
    # |-----|-----|------|-----|
    # |     |  D  |  AD  | RAD |
    # |-----|-----|------|-----|
    # |  D  | .5  |  .0  | .5  |
    # |-----|-----|------|-----|
    # | AD  | .5  |  .0  | .5  |
    # |-----|-----|------|-----|
    # | RAD | .5  |  .0  | .5  |
    # |-----|-----|------|-----|
    transition_model_key_bit0 = matrix([[.5, .0, .5], [.5, .0, .5], [.5, .0, .5]])

    # Key bit is 1
    # |-----|-----|---------|-----|
    # |     |  D  |    AD   | RAD |
    # |-----|-----|---------|-----|
    # |  D  | .0  |   1.0   | .0  |
    # |-----|-----|---------|-----|
    # | AD  | .0  |   1.0   | .0  |
    # |-----|-----|---------|-----|
    # | RAD | .0  |   1.0   | .0  |
    # |-----|-----|---------|-----|
    transition_model_key_bit1 = matrix([[.0, 1.0, .0], [.0, 1.0, .0], [.0, 1.0, .0]])

    # print "Transition model associated to key bit 0:\n", transition_model_key_bit0
    # print "Transition model associated to key bit 1:\n", transition_model_key_bit1

    models.append(transition_model_key_bit0); models.append(transition_model_key_bit1)
    return models


def singletrace_inference(belief, trace):
    return


def multitrace_inference(belief, key, trace_list):
        print "Initial belief D_0:", belief

        for trace in trace_list:
            belief = singletrace_inference(belief, trace)
