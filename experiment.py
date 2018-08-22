import cgp
import mutation
import operator
import compiledGenome
import fitness

def f0(x, y, z):
    return [x, y, z]

def f1(x, y, z):
    return [x + y * z, x * (- y), z * z * z + y - x]

def f2(x, y, z):
    return [x * y * y - z * z + y, z * z * z * z * y, x * x * (-y) * (x - z)]

functions = [operator.add, operator.neg, operator.mul, operator.sub]
functionTable = cgp.makeFunctionTable(functions)
n = 3
m = 3
h = 500
initialGenome = cgp.makeRandomGenome(n, m, h, functionTable)

a = -100
b = 100
nSamples = 50

data = fitness.makeTrainingData(f2, n, a, b, nSamples)
fitness = compiledGenome.makeCompiledFitness(data)

sigma = 1
mutationDistribution = [(2, mutation.functionMutation),(1, mutation.outputMutation),(6, mutation.connectionMutation)]
mutate = mutation.makeMutation(sigma, mutationDistribution)
nLambda = 10
goalFitness = 0.001
elite, eliteFitness = cgp.onePlusLambdaEvolve(initialGenome, fitness, mutate, nLambda, goalFitness)

cgp.printActive(elite)