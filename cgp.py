import collections
import inspect
import random
import operator
import copy
import math
import multiprocessing

FunctionRecord = collections.namedtuple("FunctionRecord", ["function", "name", "module", "arity"])

def makeFunctionTable(functions):
    #take a list of functions and construct a table with them their readable name and their arity
    functionTable = []
    for f in functions:
        spec = inspect.signature(f)
        name = dict(inspect.getmembers(f))["__name__"]
        module = dict(inspect.getmembers(f))["__module__"]
        #module = "operator"
        arity = len(spec.parameters)
        functionRecord = FunctionRecord(f, name, module, arity)
        #print(functionRecord)
        functionTable.append(functionRecord)
    return functionTable

def makeRandomGenome(n, m, h, functionTable):
    #constructs a random feed forward network with a 1xh topology
    #n: number of inputs
    #m: number of outputs
    #h: number of nodes in hidden layer

    outputs = [random.randrange(h + n)  for i in range(m)]

    nodes = []
    for i in range(h):
        #pick a random function index for this node
        fi = random.randrange(len(functionTable))
        record = functionTable[fi]
        arity = record.arity
        #generate the feed forward connections to this node
        #node i can connect to nodes 0 .. (n + i)
        connections = [random.randrange(n + i) for k in range(arity)]
        node = {'fi':fi, 'connections':connections}
        nodes.append(node)
    
    genome = {'n':n, 'm':m, 'h':h, 'functionTable':functionTable, 'nodes':nodes, 'outputs':outputs}
    return genome

def printGenome(genome):
    print(f"n:{genome['n']} m:{genome['m']} h:{genome['h']}")
    for node in genome['nodes']:
        print(f"(f{node['fi']} {node['connections']}) ", end="")
    print()
    print(genome['outputs'])

def printActive(genome):
    def evalNode(i):
        #base case: input nodes simply output their value
        if i < genome['n']:
            return f"i{i}"
        else:
            #since input nodes are not present in the node list node i is actually at position i - n
            node = genome['nodes'][i - genome['n']]
            #get the function name associated with this node
            fname = genome['functionTable'][node['fi']].name
            #calculate the paramaters to f by recursive descent on the connections
            paramaters = [evalNode(x) for x in node['connections']]
            #return the output of f applied to its paramaters
            if paramaters == []:
                result = f"({fname})"
            else:
                result = f"({fname} {' '.join(paramaters)})"
            return result
    #calculate each output
    print("\n".join([evalNode(output) for output in genome['outputs']]))

def evaluateGenome(genome, inputs):
    #calculates the output of the genome for the supplied input
    def evalNode(i):
        #base case: input nodes simply output their value
        if i < genome['n']:
            return inputs[i]
        else:
            #since input nodes are not present in the node list node i is actually at position i - n
            node = genome['nodes'][i - genome['n']]
            #get the function associated with this node
            f = genome['functionTable'][node['fi']].function
            #calculate the paramaters to f by recursive descent on the connections
            paramaters = [evalNode(x) for x in node['connections']]
            #return the output of f applied to its paramaters
            return f(*paramaters)
    #calculate each output
    return [evalNode(output) for output in genome['outputs']]

def onePlusLambdaEvolve(initialGenome, fitness, mutate, nLambda, q):
    #implements the 'one plus lambda' optimization heuristic
    #each generation, the elite is mutated lambda many times
    #then if any new genome is at least as good as the elite it becomes the new elite
    elite = initialGenome
    eliteFitness = fitness(elite)
    #generation = 0
    #the main optimization loop run forever
    while True:
        #each generation generates a new population based on the elite
        newPopulation = []
        for i in range(nLambda):
            newGenome = mutate(elite)
            newFitness = fitness(newGenome)
            newPopulation.append((newGenome, newFitness))
        #having created the new individuals we see if any are at least as good as the elite
        for individual in newPopulation:
            if individual[1] <= eliteFitness:
                elite = individual[0]
                eliteFitness = individual[1]
        q.put((eliteFitness, elite))