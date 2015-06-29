__author__ = 'umanoidTyphoon'

import collections
import copy
import math
import numpy as np
from numpy import matrix

# State related to elliptic point doubling
D = 0
# State related to elliptic point addition
AD = 1
# State related to random elliptic point addition
RAD = 2
IDHMM_IDS    = {0: 'D', 1: 'AD'}
IDHMM_STATES = {'D': 0, 'AD': 1}
#IDHMM_STATES = {'D': 0, 'AD': 1, 'RAD': 2}

REWARD = .05


class IDHMM:
    def __init__(self, key="0", trace_list=[]):
        self.key = key
        self.trace_list = trace_list

        self.belief = init_belief(self.key)
        self.init_state_distribution = init_state_distribution()

        # ---------------------------------------------- CORRECTNESS TEST ----------------------------------------------

        self.observation_model = init_observation_model_test()
        self.transition_models = init_transition_models_test()

        # ########################################### END CORRECTNESS TEST #############################################

        #self.observation_model = init_observation_model()
        #self.transition_models = init_transition_models()

    # TODO Delete it!! Inserted for debugging purposes
    def get_key(self):
        return self.key

    def get_observation_model(self):
        return self.observation_model

    def infer(self):
        guessed_key = ""
        belief = multi_trace_inference(self.belief, self.init_state_distribution, self.key, self.transition_models,
                                       self.observation_model, self.trace_list)

        print "Final belief: ", belief
        for (i,j), value in np.ndenumerate(belief):
            if value > .5:
                guessed_key += "1"
            else:
                    guessed_key += "0"
        print "Guessed key:", guessed_key

        return guessed_key

class HiddenState:
    def __init__(self, state, key_bit, prob):
        self.state = state
        self.key_bit = key_bit
        self.prob = prob

    def set_key_bit(self, key_bit):
        self.key_bit = key_bit

    def set_prob(self, prob):
        self.prob = prob

    def set_state(self, state):
        self.state = state

    def get_prob(self):
        return self.prob

    def get_key_bit(self):
        return self.key_bit

    def get_state(self):
        return self.state

    def __str__(self):
        # return "'" + self.state + "' is the current state with probability " + str(self.prob) + "."
        return "<'" + self.state + "', " + str(self.key_bit) + ", " + str(self.prob) + ">"


def init_belief(key):
    key_length = len(key)
    belief = np.zeros((1, key_length))
    belief.fill(.5)

    return belief


def init_state_distribution():
    encoded_states = IDHMM_IDS.keys()
    state_probability_distribution = np.zeros((1, len(encoded_states)))

    for state in encoded_states:
        if state == 0:
            state_probability_distribution[0, state] = 1.0
        else:
            state_probability_distribution[0, state] = 0.0

    return state_probability_distribution


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


# TODO Cambiare probabilita' a .5
def init_observation_model_test():
    # |-----|---------|---------|
    # |     |   OD    |   OAD   |
    # |-----|---------|---------|
    # |  D  |   1.0   |   0.0   |
    # |-----|---------|---------|
    # | AD  |   0.0   |   1.0   |
    # |-----|---------|---------|
    model = matrix([[1., .0], [.0, 1.]])

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


def get_ith_observation_matrix(observation_model, observation):
    ith_column = observation_model[:, IDHMM_STATES.get(observation)]
    ith_observation_matrix = copy.deepcopy(observation_model)

    for (i,j), value in np.ndenumerate(ith_observation_matrix):
        if i == j:
            ith_observation_matrix[i,j] = ith_column[i].item()
        else:
            ith_observation_matrix[i,j] = .0

    return ith_observation_matrix


def is_zero(forward_prob):
    is_zero = 1

    for (i,j), value in np.ndenumerate(forward_prob):
        if value != .0:
            is_zero = 0

    return is_zero


def normalize(forward_prob, normalization_coefficients, traceID):
    normalization_factor = .0

    for (i,j), value in np.ndenumerate(forward_prob):
        normalization_factor += math.fabs(forward_prob[i,j])
    if normalization_factor == .0:
        normalization_factor = 1.
    normalization_coefficients[0,traceID] = normalization_factor

    for (i,j), value in np.ndenumerate(forward_prob):
        forward_prob[i,j] = value / normalization_factor

    return forward_prob

def compute_beta_parm(belief, transition_models, observation_model, observation, bit_index, key_length):

    if bit_index == key_length - 1:
        return 1
    else:
        return 1
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


def clear_prob(hidden_path):
    for state in hidden_path:
        hidden_path.get(state).set_prob(.0)


def init_alpha_parm_recursion(hidden_paths, belief, observation_model, transition_models, counter, observation,
                              observation_length):
    alpha = .0
    # To remain coherent as much as possible with the paper, the initial hidden state is numbered 1.
    current_hidden_state = 1
    next_hidden_state = current_hidden_state + 1
    new_hidden_path = copy.deepcopy(hidden_paths[0])

    # DEBUG
    # print "Hidden path in the alpha initialization before clear:", print_hidden_path(new_hidden_path)
    clear_prob(new_hidden_path)
    # DEBUG
    # print "Hidden path in the alpha initialization after clear:", print_hidden_path(new_hidden_path)

    for state in IDHMM_STATES:
        for bit_value in range(2):
            p_y1_given_q1 = observation_model[IDHMM_STATES.get(state)].item(IDHMM_STATES.get(observation))
            # q0 is set to 'D'
            p_q1_given_q0_k1 = transition_models[bit_value][IDHMM_STATES.get('D')].item(IDHMM_STATES.get(state))
            p_k1 = belief[0]
            partial_p_product = p_y1_given_q1 * p_q1_given_q0_k1
            p_product = partial_p_product * p_k1

            if p_product != .0:
                # DEBUG
                print "P(y1 | q1): ", p_y1_given_q1
                print "P(q1 | q0, k1): ", p_q1_given_q0_k1
                print "P(k1): ", p_k1
                # print "Summing to alpha parm the following quantity: %f" % p_product
                updated_prob = new_hidden_path[next_hidden_state].get_prob() + p_q1_given_q0_k1
                new_hidden_path[next_hidden_state].set_state(state)
                # Set the value bit that has triggered the transition
                new_hidden_path[current_hidden_state].set_key_bit(bit_value)
                new_hidden_path[next_hidden_state].set_prob(updated_prob)
                # DEBUG
                # print "State:", (state, current_hidden_state, next_hidden_state, p_q1_given_q0_k1, p_product)
                # print "Bit value: %d" % bit_value
                # print "Hidden path: ", print_hidden_path(new_hidden_path)
                if bit_value == 0:
                    alpha = partial_p_product * (p_k1 - REWARD)
                else:
                    # alpha += p_product
                    alpha = partial_p_product * (p_k1 + REWARD)
                if (alpha > 1.0):
                    alpha = 1.0
                if (alpha < .0):
                    alpha = .0

    hidden_paths.append(new_hidden_path)
    return alpha


def compute_alpha_parm(hidden_paths, belief, transition_models, observation_model, counter, observation,
                       observation_length, bit_index, prev_alpha):
    alpha = .0
    p_yi_given_qi = .0
    p_qi_given_qprevi_ki = .0
    p_ki = .0

    hidden_set = []
    new_hidden_path = copy.deepcopy(hidden_paths[bit_index])
    for key in new_hidden_path:
        hidden_state = new_hidden_path.get(key)
        if hidden_state.get_prob() != .0:
            hidden_set.append(key)
            hidden_set.append(hidden_state.get_state())

    # DEBUG
    # print "Hidden path: ", print_hidden_path(new_hidden_path)
    # print "Hidden path before clear: ", print_hidden_path(new_hidden_path)
    clear_prob(new_hidden_path)
    # print "Hidden path after clear:  ", print_hidden_path(new_hidden_path)
    # print "Hidden set: ", print_hidden_set(hidden_set)

    for state in IDHMM_STATES:
        for bit_value in range(2):
            p_yi_given_qi = observation_model[IDHMM_STATES.get(state)].item(IDHMM_STATES.get(observation))
            # print "-----------------------------------", bit_index, state, bit_value
            for transition_state in IDHMM_STATES:
                if hidden_set.__contains__(transition_state):
                    p_qi_given_qprevi_ki = transition_models[bit_value][IDHMM_STATES.get(transition_state)].item(IDHMM_STATES.get(state))
                    print state, transition_state, bit_value, p_yi_given_qi, p_qi_given_qprevi_ki
                    # print "Hidden state:", hidden_state
                    if p_yi_given_qi != .0 and p_qi_given_qprevi_ki != .0: #\
                        # and hidden_path.get(IDHMM_STATES.get(state)).get_prob() != .0:
                        p_ki = belief[bit_index]
                        partial_p_product = p_yi_given_qi * prev_alpha * p_qi_given_qprevi_ki
                        p_product = partial_p_product * p_ki

                        print "P(yi | qi): ", p_yi_given_qi
                        print "P(qi | q(i-1), ki): ", p_qi_given_qprevi_ki
                        print "P(ki): ", p_ki
                        # print "alpha parm is equal to the following quantity: %f" % p_product

                        # prev_p_qi_given_qprevi_ki = p_qi_given_qprevi_ki
                        hidden_set_state_index = hidden_set.index(transition_state) - 1
                        hidden_state_index = hidden_set[hidden_set_state_index]
                        new_hidden_path[hidden_state_index].set_key_bit(bit_value)

                        if hidden_state_index < len(new_hidden_path):
                            updated_prob = new_hidden_path[hidden_state_index + 1].get_prob() + p_qi_given_qprevi_ki
                            new_hidden_path[hidden_state_index + 1].set_state(state)
                            # Set the value bit that has triggered the transition
                            new_hidden_path[hidden_state_index + 1].set_prob(updated_prob)
                        # print "State:", (state, hidden_state_index, p_qi_given_qprevi_ki, p_product)
                        # print "Hidden path: ", print_hidden_path(new_hidden_path)

                        #alpha += p_product
                        if bit_value == 0:
                            alpha = partial_p_product * (p_ki + REWARD)
                        else:
                            # alpha += p_product
                            alpha = partial_p_product * (p_ki + REWARD)
                        if (alpha > 1.0):
                            alpha = 1.0
                        if (alpha < .0):
                            alpha = .0

    hidden_paths.append(new_hidden_path)
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


def single_trace_inference(hidden_paths, belief, state_distribution, transition_models, observation_model, counter,
                           trace, bit, key_length):
    traceID = 0

    backward_probability_vectors = []
    forward_probability_vectors = []
    gamma_probability_vectors = []
    observations_list = trace.split()
    norm_coefficients = np.ones((1, len(observations_list)))
    skip_dictionary = dict()

    for observation in observations_list:
        for key_bit_value in range(2):
            Oi = get_ith_observation_matrix(observation_model, observation)
            alpha_T = state_distribution * transition_models[key_bit_value]
            forward_prob = alpha_T * Oi
            if is_zero(forward_prob) == 1:
                skip_list = skip_dictionary.get(observation)
                if skip_list is None:
                    skip_dictionary[observation] = [key_bit_value]
                else:
                    skip_list.append(key_bit_value)
                continue
            if key_bit_value == 0:
                for (i,j), value in np.ndenumerate(forward_prob):
                    forward_prob[i,j] = value * belief[0,traceID] * (-1)
            else:
                for (i,j), value in np.ndenumerate(forward_prob):
                    forward_prob[i,j] = value * belief[0,traceID]

            forward_prob = normalize(forward_prob, norm_coefficients, traceID)
            forward_probability_vectors.append(forward_prob)

            # print "Forward probability vector:", forward_prob

        traceID += 1
        print skip_dictionary

    print "Forward probability vectors:", forward_probability_vectors
    print "Normalization coefficient vector:", norm_coefficients

    traceID = 0
    backward_probability_vector = np.transpose(np.ones((1, len(IDHMM_STATES.keys()))))

    for observation in reversed(observations_list):
        for key_bit_value in range(2):
            skip_list = skip_dictionary.get(observation)
            if skip_list is not None and not skip_list.__contains__(key_bit_value):
                Oi = get_ith_observation_matrix(observation_model, observation)
                beta_T = copy.deepcopy(transition_models[key_bit_value])
                if key_bit_value == 0:
                    for (i,j), value in np.ndenumerate(beta_T):
                        beta_T[i,j] = value * belief[0,traceID] * -1
                else:
                    for (i,j), value in np.ndenumerate(beta_T):
                        beta_T[i,j] = value * belief[0,traceID]
                backward_prob = beta_T * Oi * backward_probability_vector

                for (i,j), value in np.ndenumerate(backward_prob):
                    backward_prob[i,j] = value / norm_coefficients[0,traceID]

                backward_probability_vectors.insert(0, np.transpose(backward_prob))
                # print "Backward probability vector:", backward_prob

        traceID += 1

    print "Backward probability vectors:", backward_probability_vectors

    vector_sizes = forward_probability_vectors[0].size

    for forward_vector,backward_vector in zip(forward_probability_vectors,backward_probability_vectors):
        gamma_vector = np.ones((1, vector_sizes))
        for (i,j), value in np.ndenumerate(gamma_vector):
            forward_component = forward_vector[i,j]
            backward_component = backward_vector[i,j]

            if forward_component < 0 and backward_component < 0:
                gamma_vector[i,j] *= -1

            gamma_vector[i,j] *= forward_component * backward_component

        gamma_vector = normalize(gamma_vector, norm_coefficients, 0)
        gamma_probability_vectors.append(gamma_vector)

    print "Gamma probability vectors:", gamma_probability_vectors

    # TODO Dividere per p(Y)!!!

    iteration = 0
    print belief
    for gamma_vector in gamma_probability_vectors:
        for (i,j), value in np.ndenumerate(gamma_vector):
            if gamma_vector[i,j] == .0:
                continue
            else:
                if gamma_vector[i,j] < 0:
                    belief[0,iteration] = .0
                else:
                    belief[0,iteration] = gamma_vector[i,j]
        iteration += 1

    print belief
    return belief


def get_key_length(observations_string):
    observations_list = observations_string.split()
    print "Observations list: ", observations_list

    return len(observations_list)


def init_hidden_paths(key_length):
    hidden_paths = []
    hidden_path  = dict()
    q0 = HiddenState('D', -1, 1.)
    hidden_path[1] = q0

    for iteration in range(2, key_length + 2):
        hidden_path[iteration] = HiddenState('Unknown', -1, 0.)

    hidden_paths.append(hidden_path)
    return hidden_paths


def multi_trace_inference(belief, state_distribution, key, transition_models, observation_model, trace_list):
    key_bit = 0
    key_length = get_key_length(trace_list[0])
    print "Supposed key length given observations: %d" % key_length

    counter = collections.Counter(trace_list)
    hidden_paths = init_hidden_paths(key_length)
    # DEBUG
    # print "Hidden Path:", print_hidden_path(hidden_paths[0])
    print "Initial belief D_0:", belief
    print "Initial state distribution S_0:", state_distribution

    for trace in trace_list:
        # DEBUG
        print "Trace under analysis:", trace
        # print "Bit number - %d" % key_bit
        print "Belief:", belief
        belief = single_trace_inference(hidden_paths, belief, state_distribution, transition_models, observation_model,
                                        counter, trace, key_bit, key_length)
        hidden_paths = init_hidden_paths(key_length)

    print "Final belief:", belief
    return belief


def print_hidden_path(hidden_path):
    to_string = "{"
    for state in hidden_path:
        to_string += str(state) + ": " + str(hidden_path.get(state)) + ", "
    to_string += "}"

    return to_string


def print_hidden_set(hidden_set):
    to_string = "("
    index = 0
    while index < len(hidden_set):
        identifier = hidden_set[index]
        hidden_state = hidden_set[index + 1]
        to_string += "<" + str(identifier) + "," + str(hidden_state) + ">, "
        index += 2
    to_string += ")"

    return to_string