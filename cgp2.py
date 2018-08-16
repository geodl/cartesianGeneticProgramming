import collections
import inspect
import random
import operator
import copy

FunctionRecord = collections.namedtuple("FunctionRecord", ["function", "arity"])

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

def go():
    def wonkyDiv(a,b):
        if b == 0:
            return math.inf
        else:
            return a / b
    functions = [operator.add, operator.neg, operator.mul, operator.sub, operator.pow, wonkyDiv]
    functionTable = makeFunctionTable(functions)
    genome = makeRandomGenome(3, 8, 10, functionTable)
    return genome

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
    return tuple([evalNode(output) for output in genome['outputs']])

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

def makeMutation(sigma, hVo, fiVc):
    def mutate(genome):
        functionTable = genome['functionTable']
        print("mutate")
        printGenome(genome)
        newGenome = copy.deepcopy(genome)
        nMutations = int(1 + abs(random.gauss(0, sigma)))
        print(f"nMutations:{nMutations}")
        for i in range(nMutations):
            #determine if we are mutating a hidden node or an output
            if random.random() < hVo:
                #we are mutating a hidden node
                print("hidden:", end="")
                nodeIndex = random.randrange(newGenome['h'])
                print(f"{nodeIndex}:", end="")
                hiddenNode = newGenome['nodes'][nodeIndex]
                #determine if we are mutating the function index or a connection
                if random.random() < fiVc:
                    print("function")
                    #we are mutating a funciton index
                    oldArity = functionTable[hiddenNode['fi']].arity
                    newFi = random.randrange(len(functionTable))
                    hiddenNode['fi'] = newFi
                    #have to maintain connections to reflect new fi
                    newArity = functionTable[newFi].arity
                    if newArity > oldArity:
                        #need to create some new connections
                        hiddenNode['connections'] += [random.randrange(nodeIndex + newGenome['n']) for i in range(newArity - oldArity)]
                    elif newArity < oldArity:
                        #need to trim some connections
                        del hiddenNode['connections'][:newArity]
                else:
                    print("connection:", end="")
                    #we are mutating a connection
                    connectionIndex = random.randrange(len(hiddenNode['connections']))
                    print(connectionIndex)
                    hiddenNode['connections'][connectionIndex] = random.randrange(nodeIndex + newGenome['n'])
            else:
                print("output:", end="")
                #we are mutating an output node
                outputIndex = random.randrange(newGenome['m'])
                print(outputIndex)
                newGenome['outputs'][outputIndex] = random.randrange(newGenome['n'] + newGenome['h'])
        printGenome(newGenome)
        return newGenome
    return mutate