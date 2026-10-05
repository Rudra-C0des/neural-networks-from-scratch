#OCT 4 2026, my first ever neural network implemented from scratch, with forward and backward propagation

#my first ever 3 layered NeuralNet :)

#we want to initialise, train (part -2: using errors to make better predictions) and query (make predictions) the NeuralNet
import numpy;
import matplotlib.pyplot;
import scipy.special;

#numpy.random.rand(m,n) => mxn wmatrix with random entres in [0,1)

class neuralNetwork: #camelCase for class names

  #inintialise the neural network
  def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
    #set the number of nodes in each input, hidden, output layer
    self.inodes = inputnodes
    self.hnodes = hiddennodes
    self.onodes = outputnodes

    #.normal means the normal distribution, 
    # numpy.random.normal thkes in 3 inputs here:
        # - 0.0 --> the center of the distribution (mean)
        # - pow(self.hnoes, -0.5) --> the standard deviation of the normal distribution
        # -  (self.hnodes, self.inodes) --> the shape of the output array, could be a number, set of numbers, etc.
        #for us it gives us a matrix of self.nodes x self.inodes numbers
        #if nothing was provided, we would have gotten 1 random number from the distribution
      #Q -> why dont we just use something like math.random() and adjust it according to the range we want, why do we want a normal distribution?
      #Ans -> we want a controlled spread keeps activationsfrom blowing up or shrinking too much, depending on the number of inputs we have, and not just random values. 

    self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)) 
    self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    #set the learing rate
    self.lr = learningrate

    #activation function is the sigmoid function
    self.activation_function = lambda x: scipy.special.expit(x) #sigmoid fn

  def train(self, inputs_list, targets_list):
    #convert inputs list to 2d array

    inputs = numpy.array(inputs_list, ndmin=2).T
    targets = numpy.array(targets_list, ndim=2).T

    #this part is almost the same as that in query...

    #calculate signals into hidden layer
    hidden_inputs = numpy.dot(self.wih, inputs)

    #calculate the singals emerging from hidden layer
    hidden_outputs = self.activation_function(hidden_inputs)

    #calculate signals into final output layer
    final_inputs = numpy.dot(self.who, hidden_outputs)

    #calculate the signals emerging from final output later
    final_inputs = numpy.dot(self.who, hidden_outputs)

    #calcuate the signals emerging grom final output layer
    final_outputs = self.activation_function(final_inputs)

    # ------- 
    #different stuff -> we need final output to find the error

    output_errors = targets - final_outputs

    #hidden layer error is the output_error split by weights, recombined at hidden nodes
    hidden_error = numpy.dot(self.who.T, output_errors) 

    #update the weights for the links between the hidden and output layers
    self.who += self.lr * numpy.dot ((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

    #update the weights for the links betweeen the input and hidden layers
    self.wih += self.lr * numpy.dot((hidden_error * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))


  def query(self, inputs_list):
    inputs = numpy.array(inputs_list, ndmin=2).T #transpose using .T , and make sure that there are atleast 2 dimensions.

    #calculate the signals into hidden layer
    hidden_inputs = numpy.dot(self.wih, inputs) #makes complete sense, simple dot product...now we do the activation function

    #caculate the signals emerging from hidden layer
    hidden_outputs = self.activation_function(hidden_inputs)

    #calcualate signals into final output layer
    final_inputs = numpy.dot(self.who, hidden_outputs)

    #calculate the singnal emerging from final output layer
    final_outputs = self.activation_function(final_inputs)


#trying to create an instance of our neuralNet

input_nodes = 3
hidden_nodes = 4
output_nodes = 3

learning_rate = 0.25

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

