__author__ = 'umanoidTyphoon'

from src.TraceGen import TraceGen

generator = TraceGen(10)
traces = generator.generate()

print "Trace list: ", traces