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
        # ---------------------------------------------- CORRECTNESS TEST ----------------------------------------------
        #
        # self.observation_model = init_observation_model_test()
        # self.transition_models = init_transition_models_test()
        #
        # ########################################### END CORRECTNESS TEST #############################################
        self.key = key
        self.trace_list = trace_list
        self.belief = self.init_belief()
        self.init_state_distribution = self.init_state_distribution()
        self.observation_model = self.init_observation_model()
        self.transition_models = self.init_transition_models()

    # TODO Delete it!! Inserted for debugging purposes
    def get_key(self):
        return self.key

    def init_belief(self):
        key_length = len(self.key)
        belief = np.zeros((1, key_length))
        belief.fill(.5)

        return belief

    def init_state_distribution(self):
        encoded_states = IDHMM_IDS.keys()
        state_probability_distribution = np.zeros((1, len(encoded_states)))

        for state in encoded_states:
            if state == 0:
                state_probability_distribution[0, state] = 1.0
            else:
                state_probability_distribution[0, state] = 0.0

        return state_probability_distribution

    def init_observation_model(self):
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

    def init_observation_model_test(self):
        # |-----|---------|---------|
        # |     |   OD    |   OAD   |
        # |-----|---------|---------|
        # |  D  |   1.0   |   0.0   |
        # |-----|---------|---------|
        # | AD  |   0.0   |   1.0   |
        # |-----|---------|---------|
        model = matrix([[1., .0], [.0, 1.]])

        return model

    def init_transition_models(self):
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

    def init_transition_models_test(self):
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

    def get_ith_observation_matrix(self, observation):
        ith_column = self.observation_model[:, IDHMM_STATES.get(observation)]
        ith_observation_matrix = np.zeros(shape=(ith_column.size,ith_column.size))

        for (i,j), value in np.ndenumerate(ith_observation_matrix):
            if i == j:
                ith_observation_matrix[i,j] = ith_column[i].item()

        return ith_observation_matrix

    def compute_alpha_parms(self, observations_list, norm_coefficients):
        forward_probability_vectors = []
        observation_ID = 0
        state_distribution = self.init_state_distribution

        # Forward step
        for observation in observations_list:
            for (i,j), state_prob in np.ndenumerate(state_distribution):
                if state_prob != .0:
                    bits_forward_probabilities = []
                    for key_bit_value in range(2):
                        Oi = self.get_ith_observation_matrix(observation)
                        alpha_T = state_distribution * self.transition_models[key_bit_value]
                        forward_prob = alpha_T * Oi
                        if key_bit_value == 0:
                            for (i,j), value in np.ndenumerate(forward_prob):
                                forward_prob[i,j] = value * (1. - self.belief[0,observation_ID])
                        else:
                            for (i,j), value in np.ndenumerate(forward_prob):
                                forward_prob[i,j] = value * self.belief[0,observation_ID]
                        bits_forward_probabilities.append(copy.deepcopy(forward_prob))

            last_forward_prob = copy.deepcopy(forward_prob)
            # Add the current forward probability to the one computed in the previous iterations
            for bits_forward_probability in bits_forward_probabilities:
                forward_prob += bits_forward_probability
            forward_prob -= last_forward_prob

            # The forward probabilities are normalized and the normalization coefficient are stored for being used in
            # the backward step
            forward_prob = normalize_and_store_coefficient(forward_prob, norm_coefficients, observation_ID)
            # forward_prob = normalize_and_store_coefficient(forward_prob, norm_coefficients, observation_ID)

            # DEBUG
            # print "State distribution before updating:", state_distribution
            # print "IDHMM decrypter :: Forward probability vector computed:", forward_prob

            # State distribution update
            state_distribution = copy.deepcopy(forward_prob)
            forward_probability_vectors.append(forward_prob)

            # Observation in the trace
            observation_ID += 1

        return forward_probability_vectors

    def compute_beta_parms(self, observations_list, norm_coefficients):
        backward_probability_vectors = []
        observation_ID = 0
        backward_probability_vector = np.transpose(np.ones((1, len(IDHMM_STATES.keys()))))

        # Backward step
        for observation in reversed(observations_list):
            for (i,j), state_prob in np.ndenumerate(backward_probability_vector):
                state_backward_probabilities = []
                if state_prob != .0:
                    Oi = self.get_ith_observation_matrix(observation)
                    bits_beta_T_transitions = []
                    for key_bit_value in range(2):
                        beta_T = copy.deepcopy(self.transition_models[key_bit_value])
                        if key_bit_value == 0:
                            for (i,j), value in np.ndenumerate(beta_T):
                                beta_T[i,j] = value * (1. - self.belief[0,observation_ID])
                        else:
                            for (i,j), value in np.ndenumerate(beta_T):
                                beta_T[i,j] = value * self.belief[0,observation_ID]
                        bits_beta_T_transitions.append(beta_T)
                    # Add the current transition matrix to the one computed in the previous iteration (since the bits are
                    # only 0 and 1)
                    beta_T += bits_beta_T_transitions[0]

                    backward_prob = beta_T * Oi * backward_probability_vector
                    for (i,j), value in np.ndenumerate(backward_prob):
                        backward_prob[i,j] = value / norm_coefficients[0,observation_ID]
                    state_backward_probabilities.append(backward_prob)

            last_backward_prob = copy.deepcopy(backward_prob)
            for state_backward_probability in state_backward_probabilities:
                backward_prob += state_backward_probability
            backward_prob -= last_backward_prob

            # DEBUG
            # print "IDHMM decrypter :: Backward probability vector:", backward_prob
            # print "State distribution before updating:", backward_probability_vector

            backward_probability_vector = copy.deepcopy(backward_prob)
            backward_probability_vectors.insert(0, np.transpose(backward_prob))

            # DEBUG
            # print "State distribution after updating:", backward_probability_vector

            observation_ID += 1

        return backward_probability_vectors

    def compute_gamma_parms(self, forward_probability_vectors,backward_probability_vectors):
        gamma_probability_vectors = []
        vector_sizes = forward_probability_vectors[0].size

        # Forward-backward step
        for forward_vector,backward_vector in zip(forward_probability_vectors,backward_probability_vectors):
            gamma_vector = np.ones((1, vector_sizes))
            for (i,j), value in np.ndenumerate(gamma_vector):
                forward_component = forward_vector[i,j]
                backward_component = backward_vector[i,j]

                gamma_vector[i,j] *= forward_component * backward_component

            # The forward probabilities are normalized
            gamma_vector = normalize(gamma_vector)
            gamma_probability_vectors.append(gamma_vector)

        return gamma_probability_vectors

    def single_trace_inference(self, hidden_paths, trace):
        observations_list = trace.split()
        norm_coefficients = np.ones((1, len(observations_list)))

        forward_probability_vectors = self.compute_alpha_parms(observations_list, norm_coefficients)
        print "IDHMM decrypter :: Forward probability vectors computed:", forward_probability_vectors

        # DEBUG
        # print "Normalization coefficient vector:", norm_coefficients

        backward_probability_vectors = self.compute_beta_parms(observations_list, norm_coefficients)

        print "IDHMM decrypter :: Backward probability vectors computed:", backward_probability_vectors

        gamma_probability_vectors = self.compute_gamma_parms(forward_probability_vectors,backward_probability_vectors)

        print "IDHMM decrypter :: Gamma probability vectors computed:", gamma_probability_vectors

        # TODO Gamma probabilities need to be divided by P(Y=y)
        iteration = 0
        # DEBUG
        # print belief

        # Update belief process
        for gamma_vector in gamma_probability_vectors:
            self.belief[0,iteration] = gamma_vector[0,IDHMM_STATES.get('AD')]
            iteration += 1

        iteration = -1

        # DEBUG
        # hidden_path = init_hidden_path(len(observations_list))
        # for hidden_path in hidden_paths:
        #      print "-----------------------------"
        #      print print_hidden_path(hidden_path)
        #      print "-----------------------------"

        hidden_states_list = []
        # Most likely hidden path in the computation
        # Start state initialization
        q0 = HiddenState('D', -1, 1.)
        hidden_states_list.append(q0)
        for gamma_vector in gamma_probability_vectors:
            iteration += 1
            most_likely_hidden_state_index = gamma_vector.argmax()
            most_likely_hidden_state_prob = np.amax(gamma_vector)
            most_likely_key_bit = -1
            if self.belief[0,iteration] <= .5:
                most_likely_key_bit = 0
            else:
                most_likely_key_bit = 1
            most_likely_previous_hidden_state = hidden_states_list[iteration]
            most_likely_previous_hidden_state.set_key_bit(most_likely_key_bit)
            most_likely_hidden_state = HiddenState(IDHMM_IDS.get(most_likely_hidden_state_index), -1,
                                                   most_likely_hidden_state_prob)
            hidden_states_list.append(most_likely_hidden_state)

        # PREVIOUS VERSION
        # print print_hidden_path(hidden_path)
        hidden_paths.append(hidden_states_list)

        return self.belief

    def multi_trace_inference(self):
        key_length = get_key_length(self.trace_list[0])
        print "\nIDHMM decrypter :: Supposed key length given observations: %d" % key_length

        counter = collections.Counter(self.trace_list)
        hidden_paths = []
        # Old version of hidden path computation
        # hidden_path = init_hidden_path(key_length)
        # hidden_paths.append(hidden_path)

        # DEBUG
        # print "Hidden Path:", print_hidden_path(hidden_paths[0])
        # print "Initial state distribution S_0:", state_distribution
        print "IDHMM decrypter :: Initial belief on the key bits:", self.belief

        for trace in self.trace_list:
            # DEBUG
            # print "Bit number - %d" % key_bit
            # print "Belief:", belief
            print "\nIDHMM decrypter :: Trace under analysis:", trace

            belief = self.single_trace_inference(hidden_paths, trace)

        print "\nIDHMM decrypter :: Hidden paths computed: "
        for hidden_path in hidden_paths:
            print print_hidden_path(hidden_path)
        # DEBUG
        # print "Final belief:", belief
        return belief

    def infer(self):
        guessed_key = ""
        belief = self.multi_trace_inference()

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
    return transition_models[key_bit][IDHMM_STATES.get(current_state),IDHMM_STATES.get(next_state)] != .0


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

# Old version of hidden path printing
# def print_hidden_path(hidden_path):
#     to_string = "{"
#     for step in hidden_path:
#         to_string += str(step) + ": ["
#         for state in hidden_path.get(step):
#              to_string += str(state) + ", "
#         to_string += "], "
#     to_string += "}"
#
#     return to_string


def print_hidden_path(hidden_path):
    iteration = 0
    path_length = len(hidden_path)
    to_string = "\t- {"

    for hidden_state in hidden_path:
        if iteration < path_length - 1:
            to_string += "q" + str(iteration) + ":[" + str(hidden_state) + "], "
        else:
            to_string += "q" + str(iteration) + ":[" + str(hidden_state) + "]"
        iteration += 1
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