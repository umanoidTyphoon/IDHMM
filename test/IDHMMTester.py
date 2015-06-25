__author__ = 'umanoidTyphoon'

from src.IDHMM import IDHMM

# idhmm = IDHMM("0101", ["AD","D","AD","D"])
# idhmm = IDHMM("11", ["AD AD", "AD AD", "AD AD", "AD AD"])
# idhmm = IDHMM("1", ["AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD"])
# idhmm = IDHMM("1011", ["AD D AD AD", "AD D AD AD", "AD D AD AD", "AD D AD AD"])
# idhmm = IDHMM("1000", ["AD D D D", "AD D D D", "AD D D D", "AD D D D"])
# idhmm = IDHMM("1001", ["AD D D AD", "AD D D AD", "AD D D AD", "AD D D AD"])
# idhmm = IDHMM("0001", ["D D D AD", "D D D AD", "D D D AD", "D D D AD"])
# idhmm = IDHMM("1", ["AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD"])
# idhmm = IDHMM("0", ["D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D", "D"])

#idhmm = IDHMM("01", ["D AD", "D AD", "D AD", "D AD", "D AD", "D AD", "D AD", "D AD"])

idhmm = IDHMM("01", ["D AD", "D AD", "D AD"])
idhmm = IDHMM("010", ["D AD D", "D AD D", "D AD D"])

idhmm.infer()