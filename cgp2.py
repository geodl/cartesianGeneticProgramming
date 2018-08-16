import collections
import inspect
import random
import operator
import copy

FunctionRecord = collections.namedtuple("FunctionRecord", ["function", "arity"])
Genome = collections.namedtuple("Genome", ["n", "m", "h", "functionTable", "nodes", "outputs"])
Node = collections.namedtuple("Node", ["functionIndex", "connections"])

def makeFunctionTable(functions):
    #take a list of functions and construct a table with them and their arity
    functionTable = []
    for f in functions:
        spec = inspect.signature(f)
        arity = len(spec.parameters)
        functionRecord = FunctionRecord(f, arity)
        functionTable.append(functionRecord)
    return functionTable

def largestConnection(i, n, r):
    #calculates the largest node index which may be connected to node i
    return (i-1 // r) * r + n

def makeRandomGenome(n, m, h, functionTable):
    #constructs a random feed forward genome with a 1xh topology
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
        node = Node(fi, connections)
        nodes.append(node)
    
    genome = Genome(n, m, h, functionTable, nodes, outputs)
    return genome

def printGenome(genome):
    print(f"n:{genome.n} m:{genome.m} h:{genome.h}")
    for node in genome.nodes:
        print(f"(f{node.functionIndex} {node.connections}) ", end="")
    print()
    print(genome.outputs)

def go():
    functions = [operator.add, operator.neg, operator.mul]
    functionTable = makeFunctionTable(functions)
    genome = makeRandomGenome(3, 8, 4, functionTable)
    return genome

def evaluateGenome(genome, inputs):
    #calculates the output of the genome for the supplied input
    def evalNode(i):
        #base case: input nodes simply output their value
        if i < genome.n:
            return inputs[i]
        else:
            #since input nodes are not present in the node list node i is actually at position i - n
            node = genome.nodes[i - genome.n]
            #get the function associated with this node
            f = genome.functionTable[node.functionIndex].function
            #calculate the paramaters to f by recursive descent on the connections
            paramaters = [evalNode(x) for x in node.connections]
            #return the output of f applied to its paramaters
            return f(*paramaters)
    #calculate each output
    return tuple([evalNode(output) for output in genome.outputs])

def onePlusLambdaEvolve(initialGenome, fitness, mutate, nLambda, nGenerations):
    #implements the 'one plus lambda' optimization heuristic
    #each generation, the elite is mutated lambda many times
    #then if any new genome is at least as good as the elite it becomes the new elite

    elite = initialGenome
    eliteFitness = fitness(elite)

    #the main optimization loop runs nGeneration many times
    for generation in range(nGenerations):
        #each generation generates a new population based on the elite
        newPopulation = []
        for i in range(nLambda):
            newGenome = mutate(elite)
            newFitness = fitness(newGenome)
            newPopulation.append((newGenome, newFitness))
        #having created the new individuals we see if any are at least as good as the elite
        for individual in newPopulation:
            if individual[1] >= eliteFitness:
                elite = individual[0]
                eliteFitness = individual[1]
    return (elite, eliteFitness)

# def makeSimpleMultiPointMutation(nodeMutationRate, outputMutationRate):
#     def mutate(genome):
#         functionTable = genome.functionTable
#         newGenome = copy.deepcopy(genome)
#         for i, node in enumerate(newGenome.nodes):
#             #possibly flip f
#             if random.random() <= nodeMutationRate:
#                 #mutate the function index
#                 newFunctionIndex = random.randrange(len(functionTable))
#                 node.functionIndex = newFunctionIndex
#                 newArity = functionTable[newFunctionIndex]
#                 if len(node.connections) > newArity:
#                     #need to trim some connections
#                     node.connections = node.connections[:newArity]
#                 elif len(node.connections) < newArity:
#                     #need to generate some new connections
#                     for j in range(newArity - len(node.connections)):
#                         upperBound = largestConnection(i, newGenome.n, newGenome.r)
#                         x = random.randrange(upperBound)
#                         node.connections.append(x)
#             #possibly flip each connection
#             for connection in node.connections:
#                 if random.random() <= nodeMutationRate:
#                     #make random connection
#             #possibly flip each output
#             for output in node.outputs:
#                 if random.random() <= outputMutationRate:
#                     #flippy output
#         return newGenome
#     return mutate