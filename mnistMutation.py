import random

def makeMacroConnectionMutation(sigma, minConnections, maxConnections):
    def mutation(genome):
        #pick hidden node
        # print(genome)
        nodeIndex = random.randrange(genome['h'])
        hiddenNode = genome['nodes'][nodeIndex]
        oldConnections = hiddenNode['connections'] 
        nConnections = len(oldConnections)
        # print(nConnections)
        k = int(nConnections + random.gauss(0, sigma))
        k = min(min(k, maxConnections), maxConnections)
        if nConnections > k:
            oldConnections = oldConnections[:k]
        elif nConnections < k:
            oldConnections += [random.randrange(nodeIndex + genome['n']) for i in range(k - nConnections)]
        # print()
        # print(k)
        # print(genome)
    return mutation

def makeMacroConnectionMutation2(minConnections, maxConnections):
    def mutation(genome):
        #pick hidden node
        # print(genome)
        nodeIndex = random.randrange(genome['h'])
        hiddenNode = genome['nodes'][nodeIndex]
        connections = hiddenNode['connections'] 
        nConnections = len(connections)
        if random.random() > .5:
            #add a connection
            if nConnections + 1 <=maxConnections:
                newConnection = random.randrange(nodeIndex + genome['n'])
                insertionPoint = random.randrange(nConnections)
                connections.insert(insertionPoint, newConnection)
        else:
            #remove a connection
            if nConnections - 1 >= minConnections:
                removalPoint = random.randrange(nConnections)
                connections.pop(removalPoint)
    return mutation