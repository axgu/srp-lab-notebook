# Feedforward Attention Summary

A feedforward neural network with attention was applied to the HCP dataset for 15-way time series classification of movie clips. Code can be found [here](ff_attention.ipynb).

Dimensions of the data were used for the following purposes:
* Movie name - apply as label for classification
* ROI - use as features for classification
* Subject - split into training and testing sets
* Time point - categorize model results as time series of accuracies

## Dataset Creation
The dataset was transformed from a dictionary with movie clip names as keys and arrays of fMRI data as values (see [summary](hcp_summary.md)) to 3-dimensional arrays of X- and Y- train and test sets. Only the first 90 seconds, equivalent to the first 90 time points, were used. 

The data was split with 100 participants for training and 76 participants for testing. A list of participants used for the testing data set was randomly generated. Participant numbers within this list had corresponding ROI feature data and movie labels stored in a testing dictionary. The same was done for participants not selected in the test set, except with a training dictionary. Dimensions for the values of these dictionaries were the same as the in the original dataset, except for 'testretest' cases, different runs from the same subject were also compiled as separate batches, converting 4 dimensions to 3.

For each dictionary, 3-dimensional arrays were created containing data across all of the first 90 time points. ROI input data was stored into an X_train or X_test array while movie labels were stored into a y_train or y_test array. ROI feature data was normalized with z-scores, and movie clip labels were one-hot encoded for each time step. Batches with less than 90 time steps were padded with 0.0. Thus, the X input sets had dimensions (batches, time steps = 90, features = 300) and the y label sets had dimensions (batches, time steps = 90, movie clips = 15).

The training datasets and testing datasets were wrapped into tensors to input into the model.

## Feedforward Neural Network
A feedforward architecture was created using PyTorch. The model consisted of an input layer, a fully connected hidden layer, an attention layer, and another fully connected output layer, as pictured below. The movie corresponding to the index of the maximum value of the 15-dimensional output vector was the predicted label. Padded values were not masked.

![](./images/ff_neural_network_attention.drawio.png)

The model was fitted using the X_train set as input and the y_train set as output across all considered time points. Time points are considered blindly, so the training set consists of 300-dimensional input vectors from all 100 train participants and across all 90 time points considered. The Adam optimizer and cross entropy loss function were used in training. Learning rate was set to 0.01 and epochs were set to 5. 

The model was then evaluated with the testing data at all 90 time points, and the accuracy was saved. Classification accuracy at each time point was plotted.

## Attention
The attention layer inputted the output from the previous fully connected hidden layer, of dimensions (batches, time steps = 90, hidden = 32) and outputted 32-dimensional context vectors for each time step of dimensions. The vectors were repeated across all batches, so the final dimensionality of the output from attention was (batches, time steps = 90, hidden = 32). Here, all time steps $1$ through $T=90$ were accessible by all other time steps.

Context vectors were calculated as the final output of the attention mechanism and expressed the relationship between hidden layer outputs for different time steps. The process below is described for an individual batch. The methods were replicated across all batches in the dataset, such that the same parameters and weight vectors/matrices were used.

A matrix $e$ of dimensions (time steps = 90, time steps = 90) was calculated. Each element $e_{ij}$ was calculated using the following equation:

$$
e_{ij} = \boldsymbol{V}_{a} ^{T} \tanh(\boldsymbol{W}_{a}h_{i} + \boldsymbol{U}_{a}h_{j}),
$$

where $h_i$ and $h_j$ are hidden layer outputs at time steps $1 \le i,j \le 90$, and $\boldsymbol{V}_{a} ^{T}$, $\boldsymbol{W}_{a}$, and $\boldsymbol{U}_{a}$ are weight matrices/parameters for the attention layer. Considering an individual batch (i.e. ignoring the first dimension), $h_i$ and $h_j$ are 32-dimensional vectors. $\boldsymbol{W}_{a}$ and $\boldsymbol{U}_{a}$ are (hidden = 32, hidden = 32) matrices; in practice, they are implemented as linear, fully connected layers with input and output dimensions as ((\*, hidden = 32), (\*, hidden = 32)). 

$\boldsymbol{V}_{a}^{T}$ is a horizontal vector of dimensions (1, hidden = 32). Performing matrix multiplication on $\boldsymbol{V}_{a}^{T}$ and the matrix from the $\tanh$ function of dimensions (hidden = 32, 1) results in scalar $e_{ij}$.

$\boldsymbol{V}_{a} ^{T}$, $\boldsymbol{W}_{a}$, and $\boldsymbol{U}_{a}$ are parameters of the feedforward model, so they are fitted through backpropagation and the loss function and optimizer used to train the other layers.

The $e$ matrix is used to calculated a matrix of attnetion scores, $\alpha$, by passing each element in $e$ through the $softmax$ function:

$$
\alpha_{ij} = \frac {\exp(e_{ij})} {\Sigma_{k=1} ^{T=90} exp(e_{ik})}
$$

Then, the context vector for each time step $1 \le i \le 90$ is calculated using (time steps = 90, time steps = 90) matrix $\alpha$ and the outputs from the hidden layer:

$$
c_{i} = \Sigma _{j=1} ^{T} \alpha_{ij} h_{j}
$$

## Random Inputs

Random inputs based on the normal distribution were generated 20 times to fill the X_test dataset of dimensions (batches, time steps = 90, features = 300). The fitted feedforward attention model was applied on each of the randomly generated datasets, and a classification accuracy was calculated. The accuracy obtained from testing the model on the original dataset and the 90th percentile of accuracies obtained from random inputs were compared as time series:

![](./images/ff_attention_accuracy.png)