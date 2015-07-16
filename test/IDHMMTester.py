__author__ = 'umanoidTyphoon'

from src.IDHMM import IDHMM
from src.KeyGen import KeyGen
from src.TraceGen import TraceGen

import csv
import datetime
import random
import sys
import time

KEY_LENGTH = 192
RANDOM_KEYS_TO_BE_TESTED = 10
MIN_TRACES_TO_BE_GENERATED = 1
MAX_TRACES_TO_BE_GENERATED = MIN_TRACES_TO_BE_GENERATED
CSV_OUTPUT_DIR_PATH = "./output"


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def writeCSV(L, error_list):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    # DEBUG
    # print error_list
    with open(CSV_OUTPUT_DIR_PATH + "/" + str(L) + "-" + timestamp, "wb") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows([error_list])


def highlight_wrong_bits(correct_key, guessed_key):
    bits_not_correctly_recovered = 0
    # Outputs the key in a better format
    sys.stdout.write("\nIDHMM tester :: Correct key - ")
    for bit in correct_key:
        sys.stdout.write(bit + " ")

    sys.stdout.write("\nIDHMM tester :: Guessed key - ")

    bit_id = 0
    for bit in guessed_key:
        if bit == correct_key[bit_id]:
            sys.stdout.write(bit + " ")
        else:
            sys.stdout.write(bcolors.FAIL + bit + bcolors.ENDC + " ")
            bits_not_correctly_recovered += 1
        bit_id += 1
    sys.stdout.write("\nIDHMM tester :: Number of key bits correctly recovered: " +
                     str(KEY_LENGTH - bits_not_correctly_recovered))
    sys.stdout.write("\nIDHMM tester :: Number of key bits NOT correctly recovered: " +
                     bcolors.FAIL + str(bits_not_correctly_recovered) + bcolors.ENDC)
    sys.stdout.write("\n")

    return bits_not_correctly_recovered


# TODO Take into account that the observations could not be sufficient to guess the password. The password has not been
# TODO guessed but the algorithm is correct
def test_correctness(idhmm):
    correctly_guessed = 0

    correct_key = idhmm.get_key()
    guessed_key = idhmm.infer()
    print "\n********************************************************************************************************" \
          "**********************************************************************************************************" \
          "**********************************************************************************************************" \
          "**************************************************************************************************\n"

    if guessed_key == correct_key:
        correctly_guessed = 1
        print "IDHMM tester :: TEST PASSED - KEY CORRECTLY GUESSED!\nIDHMM tester :: The key given in input was", \
              correct_key
    else:
        correctly_guessed = 0
        print "IDHMM tester :: TEST FAILED: KEY NOT GUESSED!\nIDHMM tester :: The key given in input was",\
               correct_key
        bits_not_correctly_recovered = highlight_wrong_bits(correct_key,guessed_key)
    print "\n********************************************************************************************************" \
          "**********************************************************************************************************" \
          "**********************************************************************************************************" \
          "**************************************************************************************************\n"

    return correctly_guessed, bits_not_correctly_recovered

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
bits_not_correctly_recovered_list = []

for i in range(RANDOM_KEYS_TO_BE_TESTED):
    keygen = KeyGen(KEY_LENGTH)
    random_binary_strings = keygen.run()
    random_traces = random.SystemRandom().randint(MIN_TRACES_TO_BE_GENERATED, MAX_TRACES_TO_BE_GENERATED)

    print "IDHMM tester :: Random binary key generated:", random_binary_strings
    print "IDHMM tester :: Number of traces that will be generated from the random binary key:", random_traces

    trace_gen = TraceGen(random_binary_strings, random_traces)
    traces = trace_gen.generate()

    print "\nIDHMM tester :: List of generated traces", traces

    idhmm = IDHMM(random_binary_strings, traces)
    guessed, bits_not_correctly_recovered = test_correctness(idhmm)
    guessed_passwords += guessed
    bits_not_correctly_recovered_list.append(bits_not_correctly_recovered)

print "IDHMM tester :: Guessed %d over %d passwords!" % (guessed_passwords, RANDOM_KEYS_TO_BE_TESTED)

writeCSV(MIN_TRACES_TO_BE_GENERATED, bits_not_correctly_recovered_list)