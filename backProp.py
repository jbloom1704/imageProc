''' Deep Hathi, Abhishek Mishra
    CSE 415, Autumn 2012
    Final Project

	Backpropogation
	Adapted from the following visuals.
	http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

	'''




import random
import math

random.seed(0)
def getWeights(rows, cols, randLow, randHigh):
	weight = []
	for i in range(rows):
			temp = []
			for j in range(cols):
				temp.append((randHigh - randLow)*random.random() +\
                                            randLow)
			weight.append(temp)
	return weight

def createNodes(nInput, nHidden, nOutput, inWeightLo, inWeightHi, outWeightLo,\
                outWeightHi):
	nInput = (nInput + 1) # 1 for bias weight
	input = [1] * nInput  
	hidden =  [1] * nHidden
	output = [1] * nOutput
	wInput = getWeights(nInput, nHidden, inWeightLo, inWeightHi)
	wOutput = getWeights(nHidden, nOutput, outWeightLo, outWeightHi)
	return [input, hidden, output, wInput, wOutput]
	

def computeNextLayerOutputs(thisLayerNodes, nextLayerNodes, thisLayerWeights):		

	for j in range(len(nextLayerNodes)):
		ac = 0.0
		for i in range(len(thisLayerNodes)):
			ac+=thisLayerWeights[i][j] * thisLayerNodes[i]	
		nextLayerNodes[j] = math.tanh(ac)
	return nextLayerNodes	
	

def computeOutputs(feature_vector, inp, hidden, output, wInput, wOutput):
	for i in range(len(inp) -1):
		inp[i] = feature_vector[i]
	
	hidden = computeNextLayerOutputs(inp, hidden, wInput)
	output = computeNextLayerOutputs(hidden, output, wOutput)
	
	return [inp, hidden, output]
	
	
		
def weightBackDrop(expectedOutputs, learning_rate, inp, hidden, output, wInput, wOutput):

	#error in output
	outputDelta = []
	for i in range(len(output)):
		error = expectedOutputs[i] - output[i]
		g_e = (1 - (output[i])**2) * error
		outputDelta.append(g_e)
	
	hiddenDelta = []		
	for i in range(len(hidden)):
		error = 0.0
		for j in range(len(output)):
			error+= outputDelta[j] * wOutput[i][j]
		g_e = (1 - (hidden[j])**2) * error
		hiddenDelta.append(g_e)
	
	for i in range(len(hidden)):
		for j in range(len(output)):
	
			dWeight = outputDelta[j] * hidden[i]
			wOutput[i][j]+= (learning_rate * dWeight)
			
	for i in range(len(inp)):
		for j in range(len(hidden)):
			dWeight = hiddenDelta[j] * inp[i]
			wInput[i][j]+=(learning_rate * dWeight)
	
	sigma_error_squared = 0.0
	for i in range(len(expectedOutputs)):
		sigma_error_squared+= 0.5 * (expectedOutputs[i] - output[i])**2
		
	return [sigma_error_squared, wInput, wOutput]	
	
	
def verify(inputSets, inp, hidden, output, wInput, wOutput):
	for inputSet in inputSets:
		inp, hidden, output = computeOutputs(inputSet[0], inp, hidden, output,\
                                                     wInput, wOutput)
		print(str(inputSet[0])+"==> "+str(output))
	return [inp, hidden, output]



	
def createAndTrain(nInput, nHidden, nOutput, inputSets, inWeightLo, inWeightHi,\
                   outWeightLo, outWeightHi, iterations, learning_rate):
	inp, hidden, output, wInput, wOutput= createNodes(nInput, nHidden, nOutput,\
                                                          inWeightLo, inWeightHi,\
                                                          outWeightLo, outWeightHi)
	for i in range(iterations):
		errorAcc = 0.0
		for inputSet in inputSets:
			feature_vector = inputSet[0]
			expectedOutputs = inputSet[1]
			inp, hidden, output = computeOutputs(feature_vector, inp,\
                                                             hidden, output, wInput,\
                                                             wOutput)
			dError, wInput, wOutput = weightBackDrop(expectedOutputs,\
                                                                 learning_rate, inp,\
                                                                 hidden, output,\
                                                                 wInput, wOutput)
			errorAcc+=dError
		if i%50 == 0:
			print("error after "+str(i)+" iterations is "+str(errorAcc))
	inp, hidden, output = verify(inputSets, inp, hidden, output, wInput, wOutput)
				

			
			
	
			
		
	
	
	







		


	
	
		
		

		
