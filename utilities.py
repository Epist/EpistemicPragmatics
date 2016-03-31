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
