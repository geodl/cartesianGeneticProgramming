import cartesianGeneticProgramming
import mutation
import operations
import mnistFitness
# import sys
import mnistMutation
import random

functions = [operations.sigmoid]
functionTable = cgp.makeFunctionTable(functions)
n = 784
m = 10
h = 40
fitness = mnistFitness.makeCompiledMNistFitness()
# fitness = mnistFitness.makeMNISTFitness()

sigma = 1

mutationDistribution = [(1, mutation.outputMutation),(12, mutation.connectionMutation), (6, mnistMutation.makeMacroConnectionMutation2(2, 40))]
mutate = mutation.makeMutation(sigma, mutationDistribution)

nLambda = 50
#never going to happen, but...
goalFitness = 0

# q1 = multiprocessing.Queue()
# q2 = multiprocessing.Queue()

# nProcesses = 1

#thought: gaussian length back connections instead of uniformly random?
def makeRandomGenome(n, m, h, functionTable, k):
    #constructs a random feed forward network with a 1xh topology
    #n: number of inputs
    #m: number of outputs
    #h: number of nodes in hidden layer

    outputs = [random.randrange(h) + n  for i in range(m)]

    nodes = []
    for i in range(h):
        #pick a random function index for this node
        fi = random.randrange(len(functionTable))
        record = functionTable[fi]
        arity = random.randrange(k) + 2
        #generate the feed forward connections to this node
        #node i can connect to nodes 0 .. (n + i)
        connections = [random.randrange(n + i) for k in range(arity)]
        node = {'fi':fi, 'connections':connections}
        nodes.append(node)
    
    genome = {'n':n, 'm':m, 'h':h, 'functionTable':functionTable, 'nodes':nodes, 'outputs':outputs}
    return genome


def experiment(q1, q2):
    initialGenome = makeRandomGenome(n, m, h, functionTable, 8)
    
    cgp.onePlusLambdaEvolve(initialGenome, fitness, mutate, nLambda, goalFitness, q1, q2)
    
if __name__ == "__main__":
    initialGenome = makeRandomGenome(n, m, h, functionTable, 8)
    cgp.onePlusLambdaEvolve(initialGenome, fitness, mutate, nLambda, goalFitness)
    # processes = [multiprocessing.Process(target=experiment, args=(q1,q2)) for i in range(nProcesses)]

    # for process in processes:
    #     process.start()
    # #hacky
    # bestFitness = 60000
    # state = "go"
    # while state == "go":
    #     eliteFitness = q1.get()
    #     if eliteFitness < bestFitness:
    #         bestFitness = eliteFitness
    #     print(1 - bestFitness/60000)
    #     if bestFitness <= goalFitness:
    #         state = "done"
    #         eliteFitness, elite = q2.get()
    #         cgp.printActive(elite)
    #         for process in processes:
    #             process.terminate()
    #             q.close()