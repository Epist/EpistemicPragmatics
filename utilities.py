#This utilities class contains the functions required to evaluate and compare models
import numpy as np
import random
from pragmodsUtils import rownorm, safelog

def entropy(dist):
    return -np.sum(dist*np.log(dist))

def mutualInformation(m1, m2):
	#Computes the mutual information between two distributions
	joint=np.outer(m1,m2)
	MI=np.sum(joint*np.log(joint/(m1*m2)))
	return MI

#Add a function that converts an RSA style lexicon to an epismteic-style lexicon

#Add a function that converts an RSA style prior distribution to an epismteic-style belief distribution
def convertPriorsToRSA(priors):
	priors=np.array(priors)
	#If a matrix of priors is given, margenalize over utterances
	#if size of dimension 2 is greater than 1 (make sure 2 is the correct dimension)
		#sum over that dimension to yield a vector
	if len(priors.shape)==2:
		priors=np.sum(priors,1)

	#Convert the prior vector (obtained either through margenaization or given by user) to a matrix
	#Do this by inserting the vector components into the matrix diagonal
	return np.matrix(np.diag(priors))

#Maybe write some code to remove the uncertainty from an epismteic model and therefore convert it into an equivalent RSA model


#Maybe write some code that allows graphical or at least intuitive editing of the matrices

def randomModel(modelSize, precision):
	#Generates a random entropy-normalized epistemic model
	#Do this by sampling random numbers for each entry and then entropy-normalizing the result
	return findBeliefShape(np.random.rand(modelSize), precision)

#def modelPerturbation(model):
	#Creates a perturbed version of a model for use as a ground truth world (an actual speaker)
	#It perturbs the model belief positions, not the belief strengths

#def modelReification(epistemicModel):
	#Creates an RSA model from an epistemic model for the purposes of model comparison
	#How well-defined is this? There is a variable and possibly significant loss here...
	#It is, however, possible to find the closest RSA model to the epistemic model by choosing lexicon entries as ones or zeros based on whether the associated probabilities are above or below the mean 


#def RSAtoEpist(rsaModel, relaxation):
	#Creates an RSA model from an epistemic model
	#Do I need to add a base-uncertainty paramater by which to transform the full confidence to approximate confidence?
	#The choice of paramater will affect the belief shape...
	#Belief shapes are not well-defined for RSA models due to their lack of uncertainty

def lexToMappingsRSAEquiv(lexicon, priors, relaxation, alpha=1):
    #Converts an RSA style lexicon to an epistemic style mapping distribution using a relaxation paramater
    #Does this by incorporating priors so that the Epistemic model reduces to the RSA model when the belief strength is 1
    mappings=np.zeros_like(lexicon, dtype='float64')
    for i, x in np.ndenumerate(lexicon):
        if x==1:
            mappings[i]=1-relaxation
        elif x==0:
            mappings[i]=relaxation
        else:
            raise ValueError("Lexicon is incorrectly specified")
    l0relaxed=(mappings*priors)#To match RSA listener-centric assumptions
    return rownorm(np.exp((alpha*safelog(rownorm(l0relaxed).T))))#To make equivalent to RSA model...
    #return rownorm(rownorm(l0relaxed).T)#To make equivalent to RSA model...

def lexToMappings(lexicon, relaxation):
	#Converts an RSA style lexicon to an epistemic style mapping distribution using a relaxation paramater
	mappings=np.zeros_like(lexicon, dtype='float64')
	for i, x in np.ndenumerate(lexicon):
		if x==1:
			mappings[i]=1-relaxation
		elif x==0:
			mappings[i]=relaxation
		else:
			raise ValueError("Lexicon is incorrectly specified")
	return rownorm(mappings.T)


def modelComparison(baseModel, model2):
	#Evaluates how similar a given model is to another model
		#This is different from getBeliefDistance in that it compares the unnormalized versions of the models.
	MIs=np.empty_like(baseModel[0])
	for i in range(np.shape(baseModel)[0]):
		MIs[i] = mutualInformation(baseModel[i,:], Model2[i,:])
	return np.mean(MIs) #Return the average mutual information

def modelEvaluation(model, world):
	#Returns the probability of a model making the correct predictions given a world (or an objective speaker)
	#Is given by KL(world||model)
	modelShape=np.shape(model)
	worldShape=np.shape(world)
	if modelShape!=worldShape:
		raise ValueError("Model and world are not of the same shape")
	else:
		kl=0
		for i, x in np.ndenumerate(world):
			kl=kl+world[i]*np.log(float(world[i])/model[i])
	return kl


def getBeliefDistance(model1, model2, precision):
	#Returns the distance between the locations of the belief distributions independent of the degree of uncertainty
	#The models may alternatively be worlds

	#As a measure for this, we use the mutual information between belief shapes (that is, entropy-normed mutual information)
	normedM1=findBeliefShape(model1, precision)
	normedM2=findBeliefShape(model2, precision)
	return modelComparison(normedM1,normedM2)



def scaleDist(dist, alpha):
    unNorm=np.power(dist, alpha)
    return rownorm(unNorm)

def findUpperBound(dist, desiredVal, bound):
    value = entropy(scaleDist(dist, bound))
    if value>desiredVal:
        newBound=np.power(bound, 2)
        return findUpperBound(dist, desiredVal, newBound)
    else:
        return bound
def binSearch(upper, lower, dist, desiredEval, precision):
    newTry=((float(upper)-lower)/2)+lower
    value=entropy(scaleDist(dist, newTry))
    #print value
    #print newTry
    if (value+precision>desiredEval) and (value-precision<desiredEval):
        #If within tolerance
        return newTry
    elif value>desiredEval:
        newLower=newTry
        return binSearch(upper, newLower, dist, desiredEval, precision)
    elif value<desiredEval:
        newUpper=newTry
        return binSearch(newUpper, lower, dist, desiredEval, presion)
    
def findBeliefStrength(dist, precision):
    #A numerical algorithm for finding the belief strength of a distribution to a given precision
    entropyNorm=np.log(np.size(dist))/2
    #precision = 0.0001
    uBound=findUpperBound(dist, entropyNorm, 2)
    beliefStrength=binSearch(uBound, 0, dist, entropyNorm, precision)
    return beliefStrength

def findBeliefShape(dist, precision):
    stren = findBeliefStrength(dist, precision)
    return scaleDist(dist, stren)


def uniformPriors(mappings):
	mapShape=np.shape(mappings)
	return np.ones(mapShape[1])/mapShape[1]


    #Need plotting and visualization code as well