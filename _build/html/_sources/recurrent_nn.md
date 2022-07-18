# Recurrent Neural Network Notes
In RNNs, hidden layers connect back to themselves, so states of hidden layers at one time point can be used as input to the hidden layers at the next time point. As a result, RNNs have a memory, which allows them to store information about past computations. 

RNNs are recurrent because they perform same computation (weights, biases) for every element in the input sequence.

## Visualization
At time point t of an RNN:

$$
o^{t} = f(h^{t}; \theta) \\
h^{t} = g(h^{t-1}, x^{t}; \theta),
$$

where $o^{t}$ is the output of the RNN at time point $t$, $x^{5}$ is the input to the RNN at time $t$, and $h^{t}$ is the state of the hidden layers at time $t$. Parameters $\theta$, the weights and biases for the network, remain the same at all time points. 

A graphical model can also be used to illustrate the relationship between variables in an RNN:

<center><img src="https://ds055uzetaobb.cloudfront.net/brioche/uploads/1Ly6D30GeF-rnn.png?width=100"></center>

This model can also be unrolled or unfolded by expanding the computation graph over time. Slices, or states of the entire RNN, are connected at each time point $t$ similar to layers in a feedforward nerual network. However, the size of an unfolded RNN depends on the size of its input and output.

<center><img src="https://ds055uzetaobb.cloudfront.net/brioche/uploads/fRVnZm2yoe-rnn_unfolded.png?width=200"></center>

## Backpropagation through Time (BPTT)
* Backpropagation algorithm applied to unrolled RNN
* 

## Vanishing/Exploding Gradients Problem
* RNNs have issues modeling long-term dependencies (i.e. relationships between elements in the sequence separated by large period of time)
* Problem arises from the gradient at early time slices being the product of many partial derivatives
    * Gradients will **vanish**, or become very small, when the partial derivatives $< 1$; learning will become very slow
    * They will **explode**, or become very large, when the partial derivates are $> 1$; learning will become unstable


## Gated Recurrent Units (GRUs)
GRUs are a form of RNN architecture that solves the vanishing gradient problem. It has an update gate and a reset gate to decide what information should be passed to the output. Both of these gates can be trained to keep information for long time periods.

### Update gate
The update gate determines the information from previous time steps that is passed along to future time steps. It is updated for time step $t$ using:

$$
z_{t} = \sigma(W^{(z)}x_{t} + U^{(z)}h_{t-1}),
$$

where $x_{t}$ is the input at $t$, $W(z)$ is the input weights, $h_{t-1}$ is the previous hidden state, and $U(z)$ is the hidden state weights.

### Reset gate
The reset gate decides how much of the past information should be forgotten. It is calculated for time step $t$ using:

$$
r_{t} = \sigma(W^{(r)}x_{t} + U^{(r)}h_{t-1})
$$

### Current memory content
A new memory content is calculated using the reset gate:

$$
h'_{t} = \tanh(Wx_{t} + r_{t}\odot Uh_{t-1})
$$

The $\odot$ operator represents the Hadamard (element-wise) product.

### Final memory at current time step
The network calculates $h_{t}$, which holds information for the current unit, using the update gate. It is updated using:

$$
h_{t} = z_{t} \odot h_{t-1} + (1-z_{t}) \odot h'_{t}
$$

## Resources
* [Recurrent Neural Networks from Brilliant](https://brilliant.org/wiki/recurrent-neural-network/)
* [GRUs from Towards Data Science](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)
