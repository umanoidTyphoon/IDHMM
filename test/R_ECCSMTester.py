__author__ = 'umanoidTyphoon'

from src.ECCSM import ECCScalarMultiplication
from src.ECCSM import RandomizedECCScalarMultiplication
from src.KeyGen import KeyGen

generator = KeyGen(2)
key = generator.run()

eccsm  = ECCScalarMultiplication(key, 1)
reccsm = RandomizedECCScalarMultiplication(key, 1)

print eccsm
print reccsm
print "ECCSM result: %d"  % eccsm.run()
print "RECCSM result: %d" % reccsm.run()
