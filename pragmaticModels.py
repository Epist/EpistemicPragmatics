#This file contains the model definitions and the code for building/specifying a particular model

import numpy as np
import utilities as util
import pragmodsUtils
from pragmodsUtils import rownorm, safelog

class PragmaticModel:
	def __init__(self,
			modelType=None,
			priors=None,
			mappings=None, # A numpy matrix with rows as utterances and columns as meanings
			meanings=None, #List of strings length |meanings|
			utterances=None, #List of strings length |utterances|
			lexicon=None,#Specify this if using an RSA model
			beliefStrength=1,
			alpha=1.0,
			maxDepth=0,#Default is 0, which does not limit recursion depth
			precision=0.0001): #Default precision for operations that require it

		self.modelType = modelType
		self.priors=np.array(priors)
		#self.mappings=np.matrix(mappings)
		self.mappings=mappings
		self.meanings=meanings
		self.utterances=utterances
		self.lexicon=lexicon
		self.beliefStrength=beliefStrength
		self.alpha=alpha
		self.maxDepth=maxDepth
		self.precision=precision

		#Set other paramaters based on the model type specified
		if self.modelType=="RSA":
			self.modelClass="RSA"
			self.explicitPriors=False #Are the nested priors explictly represented or are they to be inferred from the base prior?
		elif self.modelType=="BeliefDecay":
			self.modelClass="Epistemic"
			self.explicitPriors=False
		elif self.modelType=="ExplicitEpistemic":
			self.modelClass="Epistemic"
			self.explicitPriors=True
		#elif self.modelType=="RSAwithBeliefDecay":
			#need to define this one
		else:
			raise ValueError("Invalid model type")


		if self.priors==None:
			if self.modelClass=="Epistemic":
				self.priors=util.uniformPriors(self.mappings)
			if self.modelClass=="RSA":
				self.priors=util.uniformPriors(self.lexicon)


		#if self.modelClass=="RSA":
		#	self.priors=util.convertPriorsToRSA(self.priors)
		#else: self.priors=np.matrix(self.priors)


		#If the mappings are specified as a layer-dependent matrix, set the maxDepth
		#Equal to the number of layers that have been specified. Otherwise, set the max depth equal to infinity

		#Consider making the defaults for paramaters such as the utterance prior uniform

	def iterate(self, numIterations):
		#Iteratively compute the model predictions
		if (numIterations>self.maxDepth) and (self.maxDepth!=0):
			raise ValueError("Your iteration request exceeds the max depth of this model")
		for i in range(0, numIterations):
			if i==0:#Compute the base case
				modelState=self.computeBase()
				if self.modelClass=="Epistemic":
					interlocType="listener"
				elif self.modelClass=="RSA":
					interlocType="speaker"
			else:#Compute a higher order case
				modelState=self.computeRecursion(modelState, interlocType)
				#Now flip the interlocutor type
				if interlocType=="speaker": interlocType="listener"
				elif interlocType=="listener": interlocType="speaker"
				else: raise ValueError("Unknown interlocutor type")
		return modelState


	def computeBase(self):#Might need seperate listener and speaker functions or at least a paramater saying which one is current
		#Compute the base case model predictions
		#Returns a numpy matrix representing the current model predictions
		if self.modelClass=="RSA":
			#Compute the RSA base case as per Chris Potts' Pragmods code
			#Multiply the base probabilities with the prior and renormalize
			return rownorm(self.lexicon * self.priors) #might need to margenalize epistemic priors to get RSA priors

		if self.modelClass=="Epistemic":
			#Compute the epistemic model base case
			return self.mappings #The base case in our model is simply the mappings between meanings and utterances



	def computeRecursion(self, modelState, interlocType):
		#Compute a recursive step on top of the given model

		if self.modelClass=="RSA":
			#Compute the RSA recursive step as per Chris Potts' Pragmods code
			if interlocType=="speaker":
				#Compute the speaker recursion
				return rownorm(np.exp((self.alpha*safelog(modelState.T))))
			elif interlocType=="listener":
				#Compute the listener recursion
				return rownorm(modelState.T * self.priors)


		elif self.modelClass=="Epistemic":
			#Compute the epistemic model recursive step
			if interlocType=="speaker":
				#Compute the speaker recursion
				return rownorm(np.exp((self.alpha*safelog(modelState.T))))
			elif interlocType=="listener":
				#Compute the listener recursion
				#Matrix multiply the transposed modelState with the priors and renormalize
				return rownorm(self.beliefStrengthMod(modelState.T * self.priors))
				#return rownorm(modelState.T * self.beliefStrengthMod(self.priors))
				#return rownorm(self.beliefStrengthMod(modelState.T) * self.priors)


		#Need to check if there are explicit priors and perform something different on them

	def beliefStrengthMod(self, modelPreds):
		if modelPreds.ndim==1:
			unnorm=np.power(modelPreds, self.beliefStrength)
			return unnorm/sum(unnorm)
		else:
			return util.scaleDist(modelPreds, self.beliefStrength)
		#return rownorm(np.exp(self.beliefStrength*safelog(modelPreds)))


		#Need to work on whether decaying belief strength or fixed belief strength is the right way to perform this
		#Need to figure out whether to apply belief strength mod to priors or to full model
		#Need to get rid of global variable for current beleif strength...




	