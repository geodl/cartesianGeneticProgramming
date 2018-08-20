import cgp
import operator

def f1(x, y, z):
    return [x + y * z, x * (- y), z * z * z + y - x]

def f2(x, y, z):
    return [x, y, z]

functions = [operator.add, operator.neg, operator.mul, operator.sub]
functionTable = cgp.makeFunctionTable(functions)
n = 3
m = 3
h = 1000
initialGenome = cgp.makeRandomGenome(n, m, h, functionTable)

rangeLowerBound = -100
rangeUpperBound = 100
nSamples = 50
trainingData = cgp.makeTrainingData(f1, n, rangeLowerBound, rangeUpperBound, nSamples)
fitness = cgp.makeFitness(trainingData)

sigma = 1
mutate = cgp.makeMutation(sigma, .9, .1)

nLambda = 100
goalFitness = 0.001
elite, eliteFitness = cgp.onePlusLambdaEvolve(initialGenome, fitness, mutate, nLambda, goalFitness)

cgp.printActive(elite)