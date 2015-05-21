__author__ = 'umanoidTyphoon'

from src.ObservedECCSM import ObservedECCScalarMultiplication
from src.ObservedECCSM import ObservedRandomizedECCScalarMultiplication
from src.KeyGen import KeyGen

generator = KeyGen(5)
key = generator.run()

eccsm  = ObservedECCScalarMultiplication(key, 1)
reccsm = ObservedRandomizedECCScalarMultiplication(key, 1)

print eccsm
print reccsm

print "ECCSM result: %d"  % eccsm.run()
print "RECCSM result: %d" % reccsm.run()

print "ECCSM observations: ", eccsm.get_obs()
print "RECCSM observation: ", reccsm.get_obs()
