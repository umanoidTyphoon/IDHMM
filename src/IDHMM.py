__author__ = 'umanoidTyphoon'

import numpy as np
from numpy import matrix

# State related to elliptic point doubling
D = 0
# State related to elliptic point addition
AD = 1
# State related to random elliptic point addition
RAD = 2
IDHMM_STATES = {'D': 0, 'AD': 1}
#IDHMM_STATES = {'D': 0, 'AD': 1, 'RAD': 2}


class IDHMM:
    def __init__(self, key="0", trace_list=[]):
        self.key = key
        self.trace_list = trace_list

        self.belief = init_belief(self.key)

        # ---------------------------------------------- CORRECTNESS TEST ----------------------------------------------

        self.observation_model = init_observation_model_test()
        self.transition_models = init_transition_models_test()

        # ########################################### END CORRECTNESS TEST #############################################

        #self.observation_model = init_observation_model()
        #self.transition_models = init_transition_models()

    def get_observation_model(self):
        return self.observation_model

    def infer(self):
        belief = multitrace_inference(self.belief, self.key, self.transition_models, self.observation_model,
                                      self.trace_list)

        print "Final belief: ", belief

        for key_bit in belief:
            if key_bit > .5:
                return 2
                

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


def compute_beta_parm(belief, transition_models, observation_model, observation, bit_index):
    beta = .0
    p_ynexti_given_qnexti = .0
    p_qnexti_given_qi_knexti = .0
    p_knexti = .0

    for state in IDHMM_STATES:
        for bit_value in range(2):
            p_y1_given_q1 = observation_model[IDHMM_STATES.get(state)].item(IDHMM_STATES.get(observation))
            p_q1_given_q0_k1 = transition_models[bit_value][IDHMM_STATES.get('D')].item(IDHMM_STATES.get(state))
            p_k1 = belief[1]

            print "P(y1 | q1): ", p_y1_given_q1
            print "P(q1 | q0, k1): ", p_q1_given_q0_k1
            print "P(k1): ", p_k1
            print "Summing to beta parm the following quantity: %f" % (p_y1_given_q1 * p_q1_given_q0_k1 * p_k1)

            beta += p_y1_given_q1 * p_q1_given_q0_k1 * p_k1

    return beta

def init_alpha_parm_recursion(belief, observation_model, transition_models, observation):
    alpha = .0
    p_y1_given_q1 = .0
    p_q1_given_q0_k1 = .0
    p_k1 = .0

    for state in IDHMM_STATES:
        for bit_value in range(2):
            p_y1_given_q1 = observation_model[IDHMM_STATES.get(state)].item(IDHMM_STATES.get(observation))
            p_q1_given_q0_k1 = transition_models[bit_value][IDHMM_STATES.get('D')].item(IDHMM_STATES.get(state))
            p_k1 = belief[0]

            print "P(y1 | q1): ", p_y1_given_q1
            print "P(q1 | q0, k1): ", p_q1_given_q0_k1
            print "P(k1): ", p_k1
            print "Summing to alpha parm the following quantity: %f" % (p_y1_given_q1 * p_q1_given_q0_k1 * p_k1)

            alpha += p_y1_given_q1 * p_q1_given_q0_k1 * p_k1

    return alpha


def compute_alpha_parm(belief, transition_models, observation_model, observation, bit_index, prev_alpha):
    alpha = .0
    p_yi_given_qi = .0
    p_qi_given_qprevi_ki = .0
    p_ki = .0

    print "Observation detected: ", observation

    for state in IDHMM_STATES:
        for bit_value in range(2):
            p_yi_given_qi = observation_model[IDHMM_STATES.get(state)].item(IDHMM_STATES.get(observation))
            p_qi_given_qprevi_ki = transition_models[bit_value][IDHMM_STATES.get(state)].item(IDHMM_STATES.get(state))
            p_ki = belief[bit_index]

            print "P(yi | qi): ", p_yi_given_qi
            print "P(qi | q(i-1), ki): ", p_qi_given_qprevi_ki
            print "P(ki): ", p_ki
            print "alpha parm is equal to the following quantity: %f" % (p_yi_given_qi * prev_alpha * p_qi_given_qprevi_ki * p_ki)

            alpha += p_yi_given_qi * prev_alpha * p_qi_given_qprevi_ki * p_ki

    return alpha
#    for bit in range(get_key_length)


#    for state in IDHMM_STATES:
#        compute_alpha_parm_aux(belief, transition_models, observation_model, trace, bit_index)

    # print observation_model
    # print "-----------------------------------------"
    # print observation_model[observation].item(AD)
    # print transition_models[1][AD].item(AD)
    # print "Belief", belief[bit_index]
    #
    # p_yi_given_q_i = observation_model[AD].item(AD)
    # p_qi_given_qprev_key_bit = transition_models[1][AD].item(AD)
    # p_ki = belief.item(bit_index)
    #
    # return p_yi_given_q_i * p_qi_given_qprev_key_bit * p_ki


def singletrace_inference(belief, transition_models, observation_model, trace, bit, key_length):
    alpha_parm = None
    beta_parm  = 1.
    bayes_rule_numerator   = .0
    bayes_rule_denominator = .0
    key_bit_index = 1
    p_kn_given_yi = .0
    updated_belief = None

    observations_list = trace.split()
    first_observation = observations_list[key_bit_index - 1]
    print "Observation detected: ", first_observation

    prev_alpha = init_alpha_parm_recursion(belief, observation_model, transition_models, first_observation)

    print "Alpha initialized at %f" % prev_alpha

    print "------------------------------------------------------------------------------------------------------------"

    while key_bit_index < key_length:
        #while key_bit_index <= key_length:
        observation = observations_list[key_bit_index]
        alpha_parm = compute_alpha_parm(belief, transition_models, observation_model, observation, key_bit_index,
                                        prev_alpha)
        beta_parm  = compute_beta_parm(belief, transition_models, observation_model, observation, key_bit_index)
        p_kn_given_yi += alpha_parm * beta_parm
        belief[key_bit_index] = p_kn_given_yi
        key_bit_index += 1
    #TODO MANCA LA DIVISIONE

    # alpha_stack = []
    # beta_stack = []
    # for observation in trace:
    #     alpha_parm = compute_alpha_parm(belief, transition_models, observation_model, bit)
    #     print "Alpha parameter computed: ", alpha_parm
    #     alpha_stack.append(alpha_parm)
    #
    # while alpha_stack:
    #     bayes_rule_numerator += alpha_stack.pop()
    #
    # belief[bit] = bayes_rule_numerator
    return belief


def get_key_length(observations_string):
    observations_list = observations_string.split()
    print "Observations list: ", observations_list

    return len(observations_list)

def multitrace_inference(belief, key, transition_models, observation_model, trace_list):
    key_length = get_key_length(trace_list[0])
    print "Supposed key length given by observations: %d" % key_length

    key_bit = 0

    print "Initial belief D_0:", belief
    print observation_model

    for trace in trace_list:
        print trace
        print "Bit number - %d" % key_bit
        print "Belief:", belief
        belief = singletrace_inference(belief, transition_models, observation_model, trace, key_bit, key_length)
        key_bit += 0

    return belief