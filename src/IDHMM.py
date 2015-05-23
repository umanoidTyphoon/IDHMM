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


def init_belief(belief, key):
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
    # | AD  | .5  |  .5  | .0  |
    # |-----|-----|------|-----|
    # | RAD | .5  |  .0  | .5  |
    # |-----|-----|------|-----|


    # Key bit is 1
    # |-----|-----|------|-----|
    # |     |  D  |  AD  | RAD |
    # |-----|-----|------|-----|
    # |  D  | .5  |  .0  | .5  |
    # |-----|-----|------|-----|
    # | AD  | .5  |  .5  | .0  |
    # |-----|-----|------|-----|
    # | RAD | .5  |  .0  | .5  |
    # |-----|-----|------|-----|


def singletrace_inference(belief, trace):



def multitrace_inference(belief, key, trace_list):
        belief = init_belief(belief, key)
        print belief

        for trace in trace_list:
            belief = singletrace_inference(belief, trace)
