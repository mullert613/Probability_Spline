import numpy
import evaluate_sampled_ODE


indexes = [[0, 6],[1, 6],[2, 6],[3, 6],[4, 6],[5, 6]]
val = numpy.zeros(len(indexes))
for j in range(len(indexes)):
	val[j] = evaluate_sampled_ODE(indexes[j])