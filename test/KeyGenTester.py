__author__ = 'umanoidTyphoon'

from src.KeyGen import KeyGen

keygen = KeyGen(2)
key = keygen.run()

print "Generated random key: ", key