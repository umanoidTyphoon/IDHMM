__author__ = 'umanoidTyphoon'

from src.IDHMM import IDHMM

# idhmm = IDHMM("0101", ["AD","D","AD","D"])
idhmm = IDHMM("1", ["AD","AD","AD","AD"])

idhmm.infer()