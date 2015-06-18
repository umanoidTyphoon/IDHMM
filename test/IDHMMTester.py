__author__ = 'umanoidTyphoon'

from src.IDHMM import IDHMM

# idhmm = IDHMM("0101", ["AD","D","AD","D"])
# idhmm = IDHMM("11", ["AD AD", "AD AD", "AD AD", "AD AD"])
idhmm = IDHMM("11", ["AD AD"])

idhmm.infer()