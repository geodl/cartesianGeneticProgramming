import multiprocessing
import time
import random
import cgp
import mutation
import operations
import compiledGenome
import fitness
import sys

def f0(x, y, z):
    return [x, y, z + 6]

def f1(x, y, z):
    return [x + y * z, x * (- y), z * z * z + y - x]

def f2(x, y, z):
    return [x * y * y - z * z + y, z * z * z * z * y, x * x * (-y) * (x - z)]

def f3(x, y, z):
    return [x + 1, y + 2, z + 300]

functions = [operations.add, operations.negative, operations.multiply, operations.subtract, operations.one]
#functions = [operations.add, operations.negative, operations.multiply, operations.subtract]
functionTable = cgp.makeFunctionTable(functions)
n = 3
m = 3
h = 50
a = -100
b = 100
nSamples = 100
data = fitness.makeTrainingData(f3, n, a, b, nSamples)
fitness = compiledGenome.makeCompiledFitness(data)
sigma = 1
mutationDistribution = [(2, mutation.functionMutation),(1, mutation.outputMutation),(6, mutation.connectionMutation)]
mutate = mutation.makeMutation(sigma, mutationDistribution)
nLambda = 10
goalFitness = 0.001

q = multiprocessing.Queue()
nProcesses = 8

def experiment(q):
    initialGenome = cgp.makeRandomGenome(n, m, h, functionTable)
    cgp.onePlusLambdaEvolve(initialGenome, fitness, mutate, nLambda, q)
    
if __name__ == "__main__":
    processes = [multiprocessing.Process(target=experiment, args=(q,)) for i in range(nProcesses)]

    for process in processes:
        process.start()
    #hacky
    bestFitness = sys.float_info.max
    state = "go"
    while state == "go":
        eliteFitness, elite = q.get()
        if eliteFitness < bestFitness:
            bestFitness = eliteFitness
        print(bestFitness)
        if eliteFitness <= goalFitness:
            state = "done"
            cgp.printActive(elite)
            for process in processes:
                process.terminate()
                q.close()