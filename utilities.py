#This utilities class contains the functions required to evaluate and compare models
import numpy as np

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


def modelPerturbation(model):
	#Creates a perturbed version of a model for use as a ground truth world (an actual speaker)
	#It perturbs the model belief positions, not the belief strengths

def modelReification(epistemicModel):
	#Creates an RSA model from an epistemci model for the purposes of model comparison

def modelComparison(baseModel, model2):
	#Evaluates how well a given model

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


def getBeliefDistance(model1, model2):
	#Returns the distance between the locations of the belief distributions independent of the degree of uncertainty
	#The models may alternatively be worlds


	#A good measure for this needs to be derived. I should try it and then talk to Ed Vul for verification