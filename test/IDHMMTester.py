__author__ = 'umanoidTyphoon'

from src.IDHMM import IDHMM

# idhmm = IDHMM("0101", ["AD","D","AD","D"])
# idhmm = IDHMM("11", ["AD AD", "AD AD", "AD AD", "AD AD"])
# idhmm = IDHMM("1", ["AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD", "AD"])
idhmm = IDHMM("1111", ["AD AD AD AD", "AD AD AD AD", "AD AD AD AD", "AD AD AD AD"])
idhmm.infer()