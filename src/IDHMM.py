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
        self.belief = init_belief(self.key)
        self.init_state_distribution = init_state_distribution()
        self.observation_model = init_observation_model()
        self.transition_models = init_transition_models()

    # TODO Delete it!! Inserted for debugging purposes
    def get_key(self):
        return self.key

    def compute_alpha_parms(self, observations_list):
        forward_probability_vectors = []
        observation_ID = 0

        norm_coefficients = np.ones((1, len(observations_list)))

        # Forward step
        for observation in observations_list:
            for (i,j), state_prob in np.ndenumerate(state_distribution):
                if state_prob != .0:
                    bits_forward_probabilities = []
                    for key_bit_value in range(2):
                        Oi = get_ith_observation_matrix(self.observation_model, observation)
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
            print "IDHMM decrypter :: Forward probability vector computed:", forward_prob

            # State distribution update
            state_distribution = copy.deepcopy(forward_prob)
            forward_probability_vectors.append(forward_prob)

            # Observation in the trace
            observation_ID += 1

        return forward_probability_vectors

    def single_trace_inference(self, hidden_paths, trace):
        backward_probability_vectors = []
        forward_probability_vectors = []
        gamma_probability_vectors = []
        observations_list = trace.split()

        forward_probability_vectors = self.compute_alpha_parms(self, observations_list)

        # DEBUG
        # print "Normalization coefficient vector:", norm_coefficients
        print "IDHMM decrypter :: Forward probability vectors:", forward_probability_vectors

        # Backward step
        observation_ID = 0
        backward_probability_vector = np.transpose(np.ones((1, len(IDHMM_STATES.keys()))))

        for observation in reversed(observations_list):
            for (i,j), state_prob in np.ndenumerate(backward_probability_vector):
                state_backward_probabilities = []
                if state_prob != .0:
                    skip_list = skip_dictionary.get(observation)
                    # if skip_list is not None and not skip_list.__contains__(key_bit_value):
                    Oi = get_ith_observation_matrix(observation_model, observation)
                    bits_beta_T_transitions = []
                    for key_bit_value in range(2):
                        beta_T = copy.deepcopy(transition_models[key_bit_value])
                        # backup_beta_T = copy.deepcopy(beta_T)
                        if key_bit_value == 0:
                            for (i,j), value in np.ndenumerate(beta_T):
                                # Negative values map probabilities related to backward probabilities obtained using the bit 0
                                beta_T[i,j] = value * (1. - belief[0,observation_ID])# * (-1)
                                # backup_beta_T[i,j] = value # * (-1)
                        else:
                            for (i,j), value in np.ndenumerate(beta_T):
                                # Positive values map probabilities related to backward probabilities obtained using the bit 1
                                beta_T[i,j] = value * belief[0,observation_ID]
                                # backup_beta_T[i,j] = value
                        bits_beta_T_transitions.append(beta_T)

                    # Add the current transition matrix to the one computed in the previous iteration (since the bits are
                    # only 0 and 1)
                    beta_T += bits_beta_T_transitions[0]

                    # Backward probabilities could be negative:
                    # for (i,j), value in np.ndenumerate(backward_probability_vector):
                    # backward_probability_vector[i,j] = math.fabs(value)

                    backward_prob = beta_T * Oi * backward_probability_vector
                    # backup_backward_prob = backup_beta_T * Oi * backward_probability_vector
                    # Normalization is performed using the coefficients obtained in the forward step
                    for (i,j), value in np.ndenumerate(backward_prob):
                        if coefficients_bitmap[0,observation_ID] == 0:
                            backward_prob[i,j] = value / norm_coefficients[0,observation_ID]
                        else:
                            # backup_backward_prob[i,j] /= backup_norm_coefficients[0,observation_ID]
                            print "Else branch"
                    state_backward_probabilities.append(backward_prob)

            last_backward_prob = copy.deepcopy(backward_prob)
            for state_backward_probability in state_backward_probabilities:
                backward_prob += state_backward_probability
            backward_prob -= last_backward_prob

            backward_probability_vector = copy.deepcopy(backward_prob)
            backward_probability_vectors.insert(0, np.transpose(backward_prob))

            print "Backward probability vector:", backward_prob
            # print "Backup backward probability vector:", backup_backward_prob
            # print "State distribution after updating:", backward_probability_vector
            #     backward_probability_vector[i,j] = math.fabs(value)
            # for (i,j), value in np.ndenumerate(backward_probability_vector):
            # backward_probability_vector = copy.deepcopy(backward_prob)
            # # State distribution update
            # print "State distribution before updating:", backward_probability_vector

            observation_ID += 1

        print "Backward probability vectors:", backward_probability_vectors

        # Forward-backward step
        vector_sizes = forward_probability_vectors[0].size

        for forward_vector,backward_vector in zip(forward_probability_vectors,backward_probability_vectors):
            gamma_vector = np.ones((1, vector_sizes))
            for (i,j), value in np.ndenumerate(gamma_vector):
                forward_component = forward_vector[i,j]
                backward_component = backward_vector[i,j]

                gamma_vector[i,j] *= forward_component * backward_component

            # The forward probabilities are normalized
            gamma_vector = normalize(gamma_vector)
            gamma_probability_vectors.append(gamma_vector)

        print "Gamma probability vectors:", gamma_probability_vectors

        # TODO Gamma probabilities need to be divided by P(Y=y)
        iteration = 0
        print belief
        # Update belief process
        for gamma_vector in gamma_probability_vectors:
            belief[0,iteration] = gamma_vector[0,IDHMM_STATES.get('AD')]
            iteration += 1
        # The hidden path starts from two
        iteration = 2
        hidden_path = init_hidden_path(len(observations_list))

        # for hidden_path in hidden_paths:
        #     print "-----------------------------"
        #     print print_hidden_path(hidden_path)
        #     print "-----------------------------"

        # Hidden path computation
        for gamma_vector in gamma_probability_vectors:
            hidden_state_list = []
            previous_hidden_state_list = hidden_path.get(iteration - 1)
            for (i,j), value in np.ndenumerate(gamma_vector):
                if value == .0:
                    hidden_state_list.append(HiddenState(IDHMM_IDS.get(j), -1, math.fabs(value)))
                    continue
                key_bit_value_aux = -1
                sign = math.copysign(1, value)
                if sign < 0:
                    key_bit_value_aux = 0
                else:
                    key_bit_value_aux = 1
                for previous_hidden_state in previous_hidden_state_list:
                    if previous_hidden_state.get_prob() != .0 and \
                       transition_exist(transition_models, previous_hidden_state.get_state(), key_bit_value_aux,
                                        IDHMM_IDS.get(j)):
                        previous_hidden_state.set_key_bit(key_bit_value_aux)
                hidden_state_list.append(HiddenState(IDHMM_IDS.get(j), -1, math.fabs(value)))
            hidden_path[iteration] = hidden_state_list
            iteration += 1

        print belief
        # print print_hidden_path(hidden_path)
        # hidden_paths.append(hidden_path)
        return belief

        def multi_trace_inference(self):
            key_length = get_key_length(self.trace_list[0])
            print "IDHMM decrypter :: Supposed key length given observations - %d" % key_length

            counter = collections.Counter(self.trace_list)
            hidden_paths = []
            hidden_path  = init_hidden_path(key_length)
            hidden_paths.append(hidden_path)

            # DEBUG
            # print "Hidden Path:", print_hidden_path(hidden_paths[0])
            # print "Initial state distribution S_0:", state_distribution
            print "IDHMM decrypter :: Initial belief on the key bits:", self.belief

            for trace in self.trace_list:
                # DEBUG
                # print "Bit number - %d" % key_bit
                # print "Belief:", belief
                print "IDHMM decrypter :: Trace under analysis:", trace

                belief = single_trace_inference(hidden_paths, trace)

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