import random
import itertools
import operator
import copy

def makeMutation(sigma, mutationDistribution):
    weights = [x[0] for x in mutationDistribution]
    cumulativeWeights = list(itertools.accumulate(weights, operator.add))
    mutations = [x[1] for x in mutationDistribution]

    def selectMutation():
        return random.choices(mutations, cum_weights=cumulativeWeights)[0]
    
    def mutate(genome):
        nMutations = int(abs(random.gauss(0, sigma))) + 1
        newGenome = copy.deepcopy(genome)
        for i in range(nMutations):
            mutation = selectMutation()
            mutation(newGenome)
        return newGenome
    
    return mutate

#mutate output
def outputMutation(genome):
    outputIndex = random.randrange(genome['m'])
    genome['outputs'][outputIndex] = random.randrange(genome['n'] + genome['h'])

#mutate function
def functionMutation(genome):
    functionTable = genome['functionTable']
    nodeIndex = random.randrange(genome['h'])
    hiddenNode = genome['nodes'][nodeIndex]
    oldArity = functionTable[hiddenNode['fi']].arity
    newFi = random.randrange(len(functionTable))
    hiddenNode['fi'] = newFi
    #have to maintain connections to reflect new fi
    newArity = functionTable[newFi].arity
    if newArity > oldArity:
        #need to create some new connections
        hiddenNode['connections'] += [random.randrange(nodeIndex + genome['n']) for i in range(newArity - oldArity)]
    elif newArity < oldArity:
        #need to trim some connections
        if newArity == 0:
            hiddenNode['connections'] = []
        else:
            del hiddenNode['connections'][:newArity]

#mutate connection
def connectionMutation(genome):
    nodeIndex = random.randrange(genome['h'])
    hiddenNode = genome['nodes'][nodeIndex]
    nConnections = len(hiddenNode['connections'])
    if nConnections > 0:
        connectionIndex = random.randrange(nConnections)
        hiddenNode['connections'][connectionIndex] = random.randrange(nodeIndex + genome['n'])

#TODO: simplification mutation