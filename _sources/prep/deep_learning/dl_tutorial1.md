# Tutorial 1: Decoding Neural Responses
### Decoding vs encoding models
* Decoding - Neural activity → variable
    * How much information a brain region contains about that variable
* Encoding - Variable → neural activity
    * Approximate transformations brain performs on input variables
    * Understand how brain represents information

### Data
* Neural recordings in mice
* Two photon calcium imaging to record neural activity
* Convert imaging data into matrix of neural responses by stimuli presented
* Bin neural responses (one degree) and compute neuron’s tuning curve

### Decoding model
* Linear network with no hidden layers
* Stimulus prediction y = weights_out * neural response r + bias
* Fit network by minimizing squared error between stimulus prediction and true stimulus with loss function
* Add single hidden layer with m units to linear model
* Y = weights_out * hidden_output + bias
* Hidden layer h = weights_in * neural response r + bias
* Increasing depth and width (number of units) can increase expressivity of model - how well it can fit complex non-linear functions

### Non-linear activation functions
* Add non-linearities that allow flexible fitting
* Relu: phi(x) = max(0, x)
* Sigmoid
* Tanh
* Relu best because gradient is 1 for all x > 0 and 0 for all x < 0, so gradient can back propagate through the network as long as x > 0

### Neural network depth, width, and expressivity
* Depth - number of hidden layers
* Width - number of units in each hidden layer
* Expressivity - set of input/output transformations a network can perform, often determined by depth and width
* Cost of wider, deeper networks:
    * More parameters = more data
    * A highly parameterized network is more prone to overfitting, and requires sophisticated optimization algorithms

### Gradient descent
1. Evaluate loss on training data
2. Compute gradient of loss through backpropagation
3. Update network weights
#### Stochastic gradient descent
* Evaluate the loss at a random subset of data samples from full training set, called a mini-batch
* Bypasses the restrictive memory demands of GD by subsampling the training set into smaller mini-batches
* Adds some noise in the search for local minima of the loss function, which can help prevent overfitting

## Resources
* [Neuromatch Academy: Computational Neuroscience](https://compneuro.neuromatch.io/tutorials/W1D5_DeepLearning/student/W1D5_Tutorial1.html)
