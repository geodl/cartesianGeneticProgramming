import numpy as np
import random
import mnist
import copy
import itertools
import operator
from pathos.multiprocessing import ProcessPool
# from pathos.pp import ParallelPool

class InputNode:
    def __init__(self, nid, value=None):
        self.nid = nid
        self.value = value

    def __str__(self):
        return f"i{self.nid}"

    def output(self):
        return self.value

class HiddenNode:
    def __init__(self, nid, inputs = []):
        self.nid = nid
        self.inputs = inputs
    
    def nInputs():
        return len(self.inputs)
    
    def output(self):
        if self.memorizedOutput == None:
            activation = 0.0
            for w,n in self.inputs:
                activation += w * n.output()
            fz = 1 / (1 + np.exp(-activation))
            self.memorizedOutput = fz
            return fz
        else:
            return self.memorizedOutput
    
    def __str__(self):
        return f"h{self.nid}"

class OutputNode:
    def __init__(self, nid, value=None):
        self.nid = nid
        self.value = value
        self.memorizedOutput = None

    def __str__(self):
        return str(self.value)

    def output(self):
        if self.memorizedOutput == None:
            x = self.value.output()
            self.memorizedOutput = x
            return x
        else:
            return self.memorizedOutput

class Individual:
    def __init__(self, inputs, hidden, outputs):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        # self.compile()


    def n(self):
        return len(self.inputs)
    
    def m(self):
        return len(self.outputs)

    def h(self):
        return len(self.hidden)
    
    def __str__(self):
        si = " ".join([f"({str(i)})" for i in self.inputs])
        sh = "\n".join([f"({str(h)} {' '.join([f'({w:0.4f}, {x})' for w,x in h.inputs])})" for h in self.hidden])
        so = " ".join([f"({str(o.value)})" for o in self.outputs])
        return si + "\n" + sh + "\n" + so
    
    def setInput(self, input):
        for i,x in enumerate(input):
            self.inputs[i].value = x
        for x in self.hidden:
            x.memorizedOutput = None
        for x in self.outputs:
            x.memorizedOutput = None

    def output(self, input):
        self.setInput(input)
        return [o.output() for o in self.outputs]
    
    # def output(self, input):
    #     return self.compiled(input)
    
    # def compile(self):
    #     def evalNode(node):
    #         if type(node) is InputNode:
    #             return f"iN[{node.nid}]"
    #         elif type(node) is OutputNode:
    #             return evalNode(node.value)
    #         elif type(node) is HiddenNode:
    #             paramaters = [f"{w} * {evalNode(c)}" for w,c in node.inputs]
    #             return f"sigmoidalNode([{', '.join(paramaters)}])"

    #     ff = f'f = lambda iN:({", ".join([evalNode(o) for o in self.outputs])})'
    #     print(ff)
    #     env = {}
    #     exec(ff, globals(), env)
    #     self.compiled = env['f']

def randomNetwork(n, m, h, degreeSigma, scopeSigma):
    #make n many inputs
    networkInputs = [InputNode(i, None) for i in range(n)]
    #make h many hidden nodes
    hiddenNodes = []
    for i in range(1, h+1):
        degree = 2 + int(abs(random.gauss(0, degreeSigma)))
        inputs = []
        for j in range(degree):
            scope = 1 + int(abs(random.gauss(0, scopeSigma)))
            target = i - scope
            if target <= 0:
                #we are picking an input node
                connection = random.choice(networkInputs)
            else:
                #we are picking a hidden node
                connection = hiddenNodes[target-1]
            weight = random.random()
            nodeInput = (weight, connection)
            inputs.append(nodeInput)
        hiddenNodes.append(HiddenNode(i, inputs))
    #make m many outputs
    networkOutputs = []
    for i in range(m):
        scope = int(abs(random.gauss(0, scopeSigma)))
        target = h - scope
        if target <= 0:
            #we are picking an input node
            connection = random.choice(networkInputs)
        else:
            #we are picking a hidden node
            connection = hiddenNodes[target-1]
        networkOutputs.append(OutputNode(i, connection))
    return networkInputs, hiddenNodes, networkOutputs

def onePlusLambda(initialIndividual, fitness, mutation, λ):
    elite = initialIndividual
    eliteFitness = fitness(initialIndividual)
    generation = 0
    while True:
        newPopulation = []
        for i in range(λ):
            newGenome = mutation(elite)
            newFitness = fitness(newGenome)
            newPopulation.append((newGenome, newFitness))
        for individual in newPopulation:
            if individual[1] <= eliteFitness:
                elite = individual[0]
                eliteFitness = individual[1]
        print(f"{generation} {eliteFitness}")
        # print(str(elite))
        generation += 1



def makeFitness():
    #load
    images = mnist.train_images()
    #reshape
    images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
    #normalize
    images = np.divide(images, 255)
    #labels
    labels = mnist.train_labels()
    assert(len(images) == len(labels))

    dataset = list(zip(images, labels))

    def fitness(individual):
        errors = 0
        for image, label in dataset:
            try:
                x = individual.output(image)
            except RecursionError as re:
                print(str(individual))
            calledDigit = np.argmax(x)
            if calledDigit != label:
                errors += 1
        return errors
    return fitness

def makeFitness2():
    #load
    images = mnist.train_images()
    #reshape
    images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
    #normalize
    images = np.divide(images, 255)
    #labels
    labels = mnist.train_labels()
    assert(len(images) == len(labels))
    
    nProcesses = 4
    fullDataset = list(zip(images, labels))
    truncatedDataset = fullDataset[:-30000]
    splitDatasets = [truncatedDataset[i:i + nProcesses] for i in range(0, len(truncatedDataset), nProcesses)]
    # splitDatasets = [fullDataset[i:i + nProcesses] for i in range(0, len(fullDataset), nProcesses)]
    nDatasets = len(splitDatasets)
    pool = ProcessPool(nodes=nProcesses)

    def fitness(individual):
        def f(dataset):
            errors = 0
            for image, label in dataset:
                x = individual.output(image)
                calledDigit = np.argmax(x)
                if calledDigit != label:
                    errors += 1
            return errors
        results = pool.amap(f, splitDatasets)
        return sum(results.get())
    return fitness


def addConnection(individual):
    #choose random hidden node
    node = random.choice(individual.hidden)
    # print(node.nid)
    # target = random.randrange(node.nid + individual.n())
    target = random.randrange(node.nid)
    # print(target)
    if target == 0:
        #we are picking an input node
        connection = random.choice(individual.inputs)
    else:
        #we are picking a hidden node
        connection = individual.hidden[target - 1]
    weight = random.random()
    node.inputs.append((weight, connection))

def removeConnection(individual):
    node = random.choice(individual.hidden)
    nInputs = len(node.inputs)
    if nInputs > 2:
        i = random.randrange(nInputs)
        del node.inputs[i]

def changeConnection(individual):
    #choose random hidden node
    # i = random.randrange(individual.h())
    # print(i+1)
    # node = individual.hidden[i]
    node =  random.choice(individual.hidden)
    j = random.randrange(len(node.inputs))
    weight, _ = node.inputs[j]
    # print(node.nid)
    target = random.randrange(node.nid)
    # print(target)
    if target == 0:
        #we are picking an input node
        connection = random.choice(individual.inputs)
    else:
        #we are picking a hidden node
        connection = individual.hidden[target - 1]
    node.inputs[j] = (weight, connection)

def changeWeight(individual):
    #choose random hidden node
    i = random.randrange(individual.h())
    node = individual.hidden[i]
    j = random.randrange(len(node.inputs))
    _, connection = node.inputs[j]
    node.inputs[j] = (random.random(), connection)

def changeOutput(individual):
    i = random.randrange(individual.m())
    x = random.randrange(1 + individual.h())
    if x < 1:
        connection = random.choice(individual.inputs)
    else:
        connection = individual.hidden[x - 1]
    individual.outputs[i].value = connection

def makeMutation(nMutationsSigma, mutationDistribution):
    weights = [x[0] for x in mutationDistribution]
    cumulativeWeights = list(itertools.accumulate(weights, operator.add))
    mutations = [x[1] for x in mutationDistribution]
    
    def mutate(individual):
        nMutations = int(abs(random.gauss(0, nMutationsSigma))) + 1
        newIndividual = copy.deepcopy(individual)
        for i in range(nMutations):
            mutation = random.choices(mutations, cum_weights=cumulativeWeights)[0]
            mutation(newIndividual)
        return newIndividual
    return mutate

# def sigmoidalNode(x):
#     activation = sum(x)
#     return 1 / (1 + np.exp(-activation))



if __name__ == "__main__":
    n = 784
    m = 10
    h = 40
    degreeSigma = 3
    scopeSigma = 3

    initialIndividual = Individual(*randomNetwork(n, m, h, degreeSigma, scopeSigma))

    fitness = makeFitness2()

    nMutationsSigma = 3
    mutationDistribution = [(3, addConnection), (3, removeConnection), (5, changeConnection), (10, changeWeight), (1, changeOutput)]
    mutation = makeMutation(nMutationsSigma, mutationDistribution)

    λ = 5

    onePlusLambda(initialIndividual, fitness, mutation, λ)