import random
import cgp 

def makeTrainingData(f, n, a, b, nSamples):
    samples = []
    for i in range(nSamples):
        x = [random.uniform(a,b) for j in range(n)]
        fx = f(*x)
        samples.append((x,fx))
    return samples

def makeSimpleFitness(trainingData):
    def fitness(genome):
        #calculate error on trainingData
        error = 0.0
        for sample in trainingData:
            #x is the independant point from the domain
            x = sample[0]
            #y is the dependant point from the range
            y = sample[1]
            #fx is the approximation of the dependant point from the genome
            fx = cgp.evaluateGenome(genome, x)
            for s in range(len(y)):
                #accumulate the absolute mean error
                error += abs(y[s] - fx[s])
        return error
    return fitness