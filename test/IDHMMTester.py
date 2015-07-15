__author__ = 'umanoidTyphoon'

from src.IDHMM import IDHMM
from src.KeyGen import KeyGen
from src.TraceGen import TraceGen

import random

KEY_LENGTHS = 3
RANDOM_KEYS_TO_BE_TESTED = 10
MAX_TRACES_TO_BE_GENERATED = 4

# TODO Take into account that the observations could not be sufficient to guess the password. The password has not been
# TODO guessed but the algorithm is correct
def test_correctness(idhmm):
    correctly_guessed = 0

    guessed_key = idhmm.infer()
    print "\n********************************************************************************************************" \
          "*****************************************************************\n\n"
    if guessed_key == idhmm.get_key():
        correctly_guessed = 1
        print "TEST PASSED: KEY CORRECTLY GUESSED!\nThe key given in input was", idhmm.get_key()
    else:
        correctly_guessed = 0
        print "TEST FAILED: KEY NOT GUESSED!\nThe key given in input was", idhmm.get_key()
    print "\n\n********************************************************************************************************" \
          "*****************************************************************\n"

    return correctly_guessed

# idhmm = IDHMM("0101", ["D AD D AD", "D AD D AD", "D AD D AD", "D AD D AD"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("11", ["AD AD", "AD AD", "AD AD", "AD AD"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("1", ["AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("1011", ["AD D AD AD", "AD D AD AD", "AD D AD AD", "AD D AD AD"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("1000", ["AD D D D", "AD D D D", "AD D D D", "AD D D D"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("1001", ["AD D D AD", "AD D D AD", "AD D D AD", "AD D D AD"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("0001", ["D D D AD", "D D D AD", "D D D AD", "D D D AD"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("1", ["AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("0", ["D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("01", ["D AD", "D AD", "D AD", "D AD", "D AD", "D AD", "D AD", "D AD"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("01", ["D AD", "D AD", "D AD"])
# test_correctness(idhmm)
#
# idhmm = IDHMM("010", ["D AD D", "D AD D", "D AD D"])
# test_correctness(idhmm)
#
# print "------------------------------------------------------------------------------------------------------------\n\n"
# print "\n\n------------------------------------------------------------------------------------------------------------"
#
guessed_passwords = 0

for i in range(RANDOM_KEYS_TO_BE_TESTED):
    keygen = KeyGen(KEY_LENGTHS)
    random_binary_strings = keygen.run()
    random_traces = random.SystemRandom().randint(1, MAX_TRACES_TO_BE_GENERATED)

    print "IDHMM tester :: Random binary key generated:", random_binary_strings
    print "IDHMM tester :: Number of traces that will be generated from the random binary key:", random_traces

    trace_gen = TraceGen(random_binary_strings, random_traces)
    traces = trace_gen.generate()

    # TODO Correct the trace printing and and add IDHMM tester >> to the decrypter
    print "IDHMM tester :: List of generated traces", traces

    idhmm = IDHMM(random_binary_strings, traces)
    guessed = test_correctness(idhmm)
    guessed_passwords += guessed

print "IDHMM tester :: Guessed %d over %d passwords!" % (guessed_passwords, RANDOM_KEYS_TO_BE_TESTED)
# print "\n\n--------------------------------------------------------------------------------------------------------"

# idhmm = IDHMM("000", ['D AD D', 'D AD AD', 'D AD AD'])
#
# print test_correctness(idhmm)
#
# idhmm = IDHMM("100", ['AD D D ', 'AD D D ', 'AD D D ', 'AD D D '])
#
# print test_correctness(idhmm)

