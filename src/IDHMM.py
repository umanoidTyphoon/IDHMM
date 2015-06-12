__author__ = 'umanoidTyphoon'

import numpy as np
from numpy import matrix

# State related to elliptic point doubling
D = 0
# State related to elliptic point addition
AD = 1
# State related to random elliptic point addition
RAD = 2
IDHMM_STATES = {D, AD, RAD}


class IDHMM:
    def __init__(self, key="0", trace_list=[]):
        self.key = key
        self.trace_list = trace_list

        self.belief = init_belief(self.key)

        # ---------------------------------------------- CORRECTNESS TEST ----------------------------------------------
        self.observation_model = init_observation_model_test()
        self.transition_models = init_transition_models_test()
        # ########################################### END CORRECTNESS TEST #############################################

        self.observation_model = init_observation_model()
        self.transition_models = init_transition_models()

    def get_observation_model(self):
        return self.observation_model

    def infer(self):
        multitrace_inference(self.belief, self.key, self.transition_models, self.observation_model, self.trace_list)


def init_belief(key):
    key_length = len(key)
    belief = np.empty(key_length)
    belief.fill(.5)

    return belief


def init_observation_model():
    # |-----|---------|---------|
    # |     |   OD    |   OAD   |
    # |-----|---------|---------|
    # |  D  |   1.0   |   0.0   |
    # |-----|---------|---------|
    # | AD  |   0.0   |   0.5   |
    # |-----|---------|---------|
    # | RAD |   0.0   |   0.5   |
    # |-----|---------|---------|
    model = matrix([[1.0, .0], [.0, .5], [.0, .5]])

    return model


def init_observation_model_test():
    # |-----|---------|---------|
    # |     |   OD    |   OAD   |
    # |-----|---------|---------|
    # |  D  |   1.0   |   0.0   |
    # |-----|---------|---------|
    # | AD  |   0.0   |   1.0   |
    # |-----|---------|---------|
    model = matrix([[1.0, .0], [.0, 1.0]])

    return model


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


def init_transition_models_test():
    models = []
    # Key bit is 0
    # |-----|---------|---------|
    # |     |    D    |    AD   |
    # |-----|---------|---------|
    # |  D  |   1.0   |   .0    |
    # |-----|---------|---------|
    # | AD  |   1.0   |   .0    |
    # |-----|---------|---------|
    transition_model_key_bit0 = matrix([[1.0, .0], [1.0, .0]])

    # Key bit is 1
    # |-----|---------|---------|
    # |     |    D    |    AD   |
    # |-----|---------|---------|
    # |  D  |   .0    |   1.0   |
    # |-----|---------|---------|
    # | AD  |   .0    |   1.0   |
    # |-----|---------|---------|
    transition_model_key_bit1 = matrix([[.0, 1.0], [.0, 1.0]])

    # print "Transition model associated to key bit 0:\n", transition_model_key_bit0
    # print "Transition model associated to key bit 1:\n", transition_model_key_bit1

    models.append(transition_model_key_bit0); models.append(transition_model_key_bit1)
    return models


def compute_alpha_parm(belief, transition_models, observation_model):

    print observation_model
    print "x"
    print transition_models[1][AD].item(AD)
    print "Belief", belief.item(0)

    p_yi_given_q_i = observation_model[AD].item(AD)
    p_qi_given_qprev_key_bit = transition_models[1][AD].item(AD)
    p_ki = belief.item(0)

    return p_yi_given_q_i * p_qi_given_qprev_key_bit * p_ki

def singletrace_inference(belief, transition_models, observation_model, trace):
    alpha_parm = None
    beta_parm  = None
    bayes_rule_numerator   = None
    bayes_rule_denominator = None
    updated_belief = None

    alpha_parm = compute_alpha_parm(belief, transition_models, observation_model)

    print "Alpha parameter computed: ", alpha_parm
    return


def multitrace_inference(belief, key, transition_models, observation_model, trace_list):
        print "Initial belief D_0:", belief
        print observation_model

        for trace in trace_list:
            print trace
            belief = singletrace_inference(belief, transition_models, observation_model, trace)