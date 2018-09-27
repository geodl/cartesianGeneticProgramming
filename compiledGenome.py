import cgp
import operations
import fitness

def compileGenome(genome):
    def evalNode(i):
        #base case: input nodes simply output their value
        if i < genome['n']:
            return f"iN[{i}]"
            # return f"iN[{i}]"
        else:
            #since input nodes are not present in the node list node i is actually at position i - n
            node = genome['nodes'][i - genome['n']]
            #get the function name associated with this node
            fname = genome['functionTable'][node['fi']].name
            fmodule = genome['functionTable'][node['fi']].module

            #calculate the paramaters to f by recursive descent on the connections
            paramaters = [evalNode(x) for x in node['connections']]
            #return the output of f applied to its paramaters
            signature = f"{fmodule + '.' + fname}({', '.join(paramaters)})"
            # signature = f"{fmodule + '.' + fname}({paramaters})"
            #signature = f"{fname}({', '.join(paramaters)})"

            return signature
    # header = "f = lambda " + ', '.join(['i' + str(i) for i in range(genome['n'])]) + ":"
    header = "f = lambda iN:"

    body = f'({", ".join([evalNode(output) for output in genome["outputs"]])})'
    #body = f'{[evalNode(output) for output in genome["outputs"]]}'
   # body = f'{[evalNode(output) for output in genome["outputs"]]}'

    ff = header + body
    #env = {'operations':operations}
    env = {}
    # print(ff)
    exec(ff, globals(), env)
    f = env['f']
    return f

def makeCompiledFitness(trainingData):
    def compiledFitness(genome):
        genomeFunction = compileGenome(genome)
        error = 0.0
        for sample in trainingData:
            #x is the independant point from the domain
            x = sample[0]
            #y is the dependant point from the range
            y = sample[1]
            #fx is the approximation of the dependant point from the genome
            fx = genomeFunction(x)
            for s in range(len(y)):
                #accumulate the absolute mean error
                error += abs(y[s] - fx[s])
        return error
    return compiledFitness