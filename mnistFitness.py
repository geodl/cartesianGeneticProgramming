import mnist
import numpy as np
import cgp
import compiledGenome

def makeMNISTFitness():
    #load
    images = mnist.train_images()
    #reshape
    images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
    #normalize
    images = np.divide(images, 255)
    #labels
    labels = mnist.train_labels()
    def fitness(genome):
        error = 0
        for i, image in enumerate(images):
            fx = cgp.evaluateGenome(genome, image)
            #find largest output
            x = 0.0
            xi = 0
            for j, output in enumerate(fx):
                if output > x:
                    x = output
                    xi = j
            if xi != labels[i]:
                error += 1
        return error
    return fitness

def makeCompiledMNistFitness():
    #load
    images = mnist.train_images()
    #reshape
    images = images.reshape((images.shape[0], images.shape[1] * images.shape[2]))
    #normalize
    images = np.divide(images, 255)
    #labels
    labels = mnist.train_labels()
    def compiledFitness(genome):
        genomeFunction = compiledGenome.compileGenome(genome)
        error = 0
        for i, image in enumerate(images):
            genomeOutput = genomeFunction(image)
            #find largest output which is the digit called by the genome
            calledDigit = np.argmax(genomeOutput)
            # print(labels[i])
            # print(genomeOutput)
            if calledDigit != labels[i]:
                error += 1
        return error
    return compiledFitness