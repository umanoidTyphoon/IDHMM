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

# States used for testing the correctness
# IDHMM_STATES = {'D': 0, 'AD': 1}

# Input Driven Hidden Markov Model identifiers associated to the hidden states
IDHMM_IDS = {0: 'D', 1: 'AD', 2: 'RAD'}
# Input Driven Hidden Markov Model states associated to their identifiers
IDHMM_STATES = {'D': 0, 'AD': 1, 'RAD': 2}


class IDHMM:
    def __init__(self, key="0", trace_list=[]):
        self.key = key
        self.trace_list = trace_list
        self.belief = init_belief(self.key)
        self.init_state_distribution = init_state_distribution()
        self.observation_model = init_observation_model()
        self.transition_models = init_transition_models()

        # ---------------------------------------------- CORRECTNESS TEST ----------------------------------------------
        #
        # self.observation_model = init_observation_model_test()
        # self.transition_models = init_transition_models_test()
        #
        # ########################################### END CORRECTNESS TEST #############################################

    # TODO Delete it!! Inserted for debugging purposes
    def get_key(self):
        return self.key

    def get_observation_model(self):
        return self.observation_model

    def infer(self):
        guessed_key = ""
        belief = multi_trace_inference(self.belief, self.init_state_distribution, self.transition_models,
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
    ith_observation_matrix = np.zeros(shape=(ith_column.size,ith_column.size))

    for (i,j), value in np.ndenumerate(ith_observation_matrix):
        if i == j:
            ith_observation_matrix[i,j] = ith_column[i].item()

    return ith_observation_matrix


def is_zero(prob_vector):
    is_zero = 1

    for (i,j), value in np.ndenumerate(prob_vector):
        if value != .0:
            is_zero = 0

    return is_zero


def normalize_and_store_coefficient(prob_vector, normalization_coefficients, observation_ID):
    normalization_factor = .0

    for (i,j), value in np.ndenumerate(prob_vector):
        normalization_factor += math.fabs(prob_vector[i,j])
    if normalization_factor == .0:
        normalization_factor = 1.
    normalization_coefficients[0,observation_ID] = normalization_factor

    for (i,j), value in np.ndenumerate(prob_vector):
        prob_vector[i,j] = value / normalization_factor

    return prob_vector


def normalize(prob_vector):
    normalization_factor = .0

    for (i,j), value in np.ndenumerate(prob_vector):
        normalization_factor += math.fabs(prob_vector[i,j])
    if normalization_factor == .0:
        normalization_factor = 1.
    for (i,j), value in np.ndenumerate(prob_vector):
        prob_vector[i,j] = value / normalization_factor

    return prob_vector


def transition_exist(transition_models, current_state, key_bit, next_state):
    # if (transition_models[key_bit][IDHMM_STATES.get(current_state),IDHMM_STATES.get(next_state)] != .0) == False:
    #    print "The check is false!"
    return transition_models[key_bit][IDHMM_STATES.get(current_state),IDHMM_STATES.get(next_state)] != .0


def single_trace_inference(hidden_paths, belief, state_distribution, transition_models, observation_model, trace):
    backward_probability_vectors = []
    forward_probability_vectors = []
    gamma_probability_vectors = []
    observations_list = trace.split()
    skip_dictionary = dict()
    observation_ID = 0

    # Forward step
    coefficients_bitmap = np.zeros((1, len(observations_list)))
    norm_coefficients = np.ones((1, len(observations_list)))
    backup_norm_coefficients = np.ones((1, len(observations_list)))
    for observation in observations_list:
        for key_bit_value in range(2):
            Oi = get_ith_observation_matrix(observation_model, observation)
            alpha_T = state_distribution * transition_models[key_bit_value]
            forward_prob = alpha_T * Oi
            # Used when the forward probability is multiplied by the belief and the result is equal to 0
            backup_forward_prob = copy.deepcopy(forward_prob)
            # The skip dictionary tracks the <key,value> pairs, which represents the transitions having 0 probabilities
            # to occur, given the observation. This skip dictionary is queried in the backward step, to avoid unuseful
            # iterations
            if is_zero(forward_prob) == 1:
                skip_list = skip_dictionary.get(observation)
                if skip_list is None:
                    skip_dictionary[observation] = [key_bit_value]
                else:
                    skip_list.append(key_bit_value)
                continue
            if key_bit_value == 0:
                for (i,j), value in np.ndenumerate(forward_prob):
                    # Negative values map probabilities related to forward probabilities obtained using the bit 0
                    forward_prob[i,j] = math.fabs(value) * belief[0,observation_ID] * (-1)
                    backup_forward_prob[i,j] = math.fabs(value) * (-1)
            else:
                for (i,j), value in np.ndenumerate(forward_prob):
                    # Positive values map probabilities related to forward probabilities obtained using the bit 1
                    forward_prob[i,j] = math.fabs(value) * belief[0,observation_ID]
                    backup_forward_prob[i,j] = math.fabs(value)

            # The forward probabilities are normalized and the normalization coefficient are stored for being used in
            # the backward step
            forward_prob = normalize_and_store_coefficient(forward_prob, norm_coefficients, observation_ID)
            backup_forward_prob = normalize_and_store_coefficient(backup_forward_prob, backup_norm_coefficients,
                                                                  observation_ID)

            print "Forward probability vector:", forward_prob
            print "Backup forward probability vector:", backup_forward_prob
            # print "State distribution before updating:", state_distribution
            # State distribution update
            if is_zero(forward_prob) == 1:
                # The forward probability vector is constituted by only 0s. It cannot be used to update the state
                # distribution. For optimization reasons, the forward probability vector list is update accordingly
                state_distribution = copy.deepcopy(backup_forward_prob)
                forward_probability_vectors.append(backup_forward_prob)
                # For the reason expressed above, this operation allows to know which normalization coefficient has to
                # be use in the next normalization
                coefficients_bitmap[0,observation_ID] = 1
            else:
                state_distribution = copy.deepcopy(forward_prob)
                forward_probability_vectors.append(forward_prob)
                #for (i,j), value in np.ndenumerate(state_distribution):
                #state_distribution[i,j] = math.fabs(value)
                # print "State distribution after updating:", state_distribution

        # Observation in the trace
        observation_ID += 1
        print skip_dictionary

    print "Forward probability vectors:", forward_probability_vectors
    print "Normalization coefficient vector:", norm_coefficients
    print "Backup normalization coefficient vector:", backup_norm_coefficients
    print "Coefficient bitmap:", coefficients_bitmap

    # Backward step
    # observation_ID = 0
    # backward_probability_vector = np.transpose(np.ones((1, len(IDHMM_STATES.keys()))))
    #
    # for observation in reversed(observations_list):
    #     for key_bit_value in range(2):
    #         skip_list = skip_dictionary.get(observation)
    #         if skip_list is not None and not skip_list.__contains__(key_bit_value):
    #             Oi = get_ith_observation_matrix(observation_model, observation)
    #             beta_T = copy.deepcopy(transition_models[key_bit_value])
    #             backup_beta_T = copy.deepcopy(beta_T)
    #             if key_bit_value == 0:
    #                 for (i,j), value in np.ndenumerate(beta_T):
    #                     # Negative values map probabilities related to backward probabilities obtained using the bit 0
    #                     beta_T[i,j] = math.fabs(value) * belief[0,observation_ID] * (-1)
    #                     backup_beta_T[i,j] = math.fabs(value) * (-1)
    #             else:
    #                 for (i,j), value in np.ndenumerate(beta_T):
    #                     # Positive values map probabilities related to backward probabilities obtained using the bit 1
    #                     beta_T[i,j] = math.fabs(value) * belief[0,observation_ID]
    #                     backup_beta_T[i,j] = math.fabs(value)
    #
    #             # Backward probabilities could be negative:
    #             for (i,j), value in np.ndenumerate(backward_probability_vector):
    #                 backward_probability_vector[i,j] = math.fabs(value)
    #
    #             backward_prob = beta_T * Oi * backward_probability_vector
    #             backup_backward_prob = backup_beta_T * Oi * backward_probability_vector
    #
    #             # Normalization is performed using the coefficients obtained in the forward step
    #             for (i,j), value in np.ndenumerate(backward_prob):
    #                 if coefficients_bitmap[0,observation_ID] == 0:
    #                     backward_prob[i,j] = value / norm_coefficients[0,observation_ID]
    #                 else:
    #                     backup_backward_prob[i,j] /= backup_norm_coefficients[0,observation_ID]
    #
    #             if is_zero(backward_prob) == 1:
    #                 # The backward probability vector is constituted by only 0s. A similar reasoning to the one applied
    #                 # for the forward step is applied here
    #                 backward_probability_vector = copy.deepcopy(backup_backward_prob)
    #                 backward_probability_vectors.insert(0, np.transpose(backup_backward_prob))
    #             else:
    #                 backward_probability_vector = copy.deepcopy(backward_prob)
    #                 backward_probability_vectors.insert(0, np.transpose(backward_prob))
    #
    #             print "Backward probability vector:", backward_prob
    #             print "Backup backward probability vector:", backup_backward_prob
    #             # print "State distribution after updating:", backward_probability_vector
    #             #     backward_probability_vector[i,j] = math.fabs(value)
    #             # for (i,j), value in np.ndenumerate(backward_probability_vector):
    #             # backward_probability_vector = copy.deepcopy(backward_prob)
    #             # # State distribution update
    #             # print "State distribution before updating:", backward_probability_vector
    #
    #
    #     observation_ID += 1
    #
    # print "Backward probability vectors:", backward_probability_vectors
    #
    # # Forward-backward step
    # vector_sizes = forward_probability_vectors[0].size
    #
    # for forward_vector,backward_vector in zip(forward_probability_vectors,backward_probability_vectors):
    #     gamma_vector = np.ones((1, vector_sizes))
    #     for (i,j), value in np.ndenumerate(gamma_vector):
    #         forward_component = forward_vector[i,j]
    #         backward_component = backward_vector[i,j]
    #         # Negative values map probabilities related to gamma probabilities
    #         if forward_component < 0 and backward_component < 0:
    #             gamma_vector[i,j] *= -1
    #
    #         gamma_vector[i,j] *= forward_component * backward_component
    #
    #     # The forward probabilities are normalized
    #     gamma_vector = normalize(gamma_vector)
    #     gamma_probability_vectors.append(gamma_vector)
    #
    # print "Gamma probability vectors:", gamma_probability_vectors
    #
    # # TODO Gamma probabilities need to be divided by P(Y=y)
    # iteration = 0
    # print belief
    # # Update belief process
    # for gamma_vector in gamma_probability_vectors:
    #     for (i,j), value in np.ndenumerate(gamma_vector):
    #         if gamma_vector[i,j] == .0:
    #             continue
    #         else:
    #             if gamma_vector[i,j] < 0:
    #                 belief[0,iteration] = .0
    #             else:
    #                 belief[0,iteration] = gamma_vector[i,j]
    #     iteration += 1
    # # The hidden path starts from two
    # iteration = 2
    # hidden_path = init_hidden_path(len(observations_list))
    #
    # # for hidden_path in hidden_paths:
    # #     print "-----------------------------"
    # #     print print_hidden_path(hidden_path)
    # #     print "-----------------------------"
    #
    # # Hidden path computation
    # for gamma_vector in gamma_probability_vectors:
    #     hidden_state_list = []
    #     previous_hidden_state_list = hidden_path.get(iteration - 1)
    #     for (i,j), value in np.ndenumerate(gamma_vector):
    #         if value == .0:
    #             hidden_state_list.append(HiddenState(IDHMM_IDS.get(j), -1, math.fabs(value)))
    #             continue
    #         key_bit_value_aux = -1
    #         sign = math.copysign(1, value)
    #         if sign < 0:
    #             key_bit_value_aux = 0
    #         else:
    #             key_bit_value_aux = 1
    #         for previous_hidden_state in previous_hidden_state_list:
    #             if previous_hidden_state.get_prob() != .0 and \
    #                transition_exist(transition_models, previous_hidden_state.get_state(), key_bit_value_aux,
    #                                 IDHMM_IDS.get(j)):
    #                 previous_hidden_state.set_key_bit(key_bit_value_aux)
    #         hidden_state_list.append(HiddenState(IDHMM_IDS.get(j), -1, math.fabs(value)))
    #     hidden_path[iteration] = hidden_state_list
    #     iteration += 1

    print belief
    # print print_hidden_path(hidden_path)
    # hidden_paths.append(hidden_path)
    return belief


def get_key_length(observations_string):
    observations_list = observations_string.split()
    print "Observations list: ", observations_list

    return len(observations_list)


def init_hidden_path(key_length):
    hidden_path  = dict()
    q0D   = HiddenState('D', -1, 1.)
    q0AD  = HiddenState('AD', -1, 0.)
    q0RAD = HiddenState('RAD', -1, 0.)

    hidden_path[1] = [q0D,q0AD,q0RAD]

    for iteration in range(2, key_length + 2):
        hidden_state_list = [HiddenState('Unknown', -1, 0.), HiddenState('Unknown', -1, 0.), HiddenState('Unknown', -1, 0.)]
        hidden_path[iteration] = hidden_state_list

    return hidden_path


def multi_trace_inference(belief, state_distribution, transition_models, observation_model, trace_list):
    key_length = get_key_length(trace_list[0])
    print "Supposed key length given observations: %d" % key_length

    counter = collections.Counter(trace_list)
    hidden_paths = []
    hidden_path  = init_hidden_path(key_length)
    hidden_paths.append(hidden_path)

    # DEBUG
    print "Hidden Path:", print_hidden_path(hidden_paths[0])
    print "Initial belief D_0:", belief
    print "Initial state distribution S_0:", state_distribution

    for trace in trace_list:
        # DEBUG
        print "Trace under analysis:", trace
        # print "Bit number - %d" % key_bit
        print "Belief:", belief
        belief = single_trace_inference(hidden_paths, belief, state_distribution, transition_models, observation_model,
                                        trace)
        for hidden_path in hidden_paths:
            print print_hidden_path(hidden_path)
    print "Final belief:", belief
    return belief


def print_hidden_path(hidden_path):
    to_string = "{"
    for step in hidden_path:
        to_string += str(step) + ": ["
        for state in hidden_path.get(step):
             to_string += str(state) + ", "
        to_string += "], "
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