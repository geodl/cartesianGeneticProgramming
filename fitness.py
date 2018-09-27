import random
import cgp 

def makeSImpleTrainingData(f, n, a, b, nSamples):
    samples = []
    for i in range(nSamples):
        x = [random.uniform(a,b) for j in range(n)]
        fx = f(*x)
        samples.append((x,fx))
    return samples



def makeSimpleFitness(labledData, compile):
    def fitness(genome):
        #calculate error on trainingData
        error = 0.0
        for sample in labledData:
            #x is the independant point from the domain
            x = sample[0]
            #label is the labled dependant point from the range
            label = sample[1]
            #fx is the approximation of the dependant point from the genome
            if compile:
                genomeFunction = compiledGenome.compileGenome(genome)
                fx = genomeFunction(x)
            else:
                fx = cgp.evaluateGenome(genome, x)
            for s in range(len(y)):
                #accumulate the absolute mean error
                error += abs(y[s] - fx[s])
        return error
    return fitness