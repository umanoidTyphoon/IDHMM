__author__ = 'umanoidTyphoon'

import numpy as np

class IDHMM:
    def __init__(self, key, trace_list):
        self.belief = []
        self.key = key
        self.trace_list = trace_list

    def infer(self):
        multitrace_inference(self.belief, self.key, self.trace_list)


def init_belief(belief, key):
    key_length = len(key)
    belief = np.empty(key_length)
    belief.fill(.5)

    return belief


def singletrace_inference(belief, trace):



def multitrace_inference(belief, key, trace_list):
        belief = init_belief(belief, key)
        print belief

        for trace in trace_list:
            belief = singletrace_inference(belief, trace)
