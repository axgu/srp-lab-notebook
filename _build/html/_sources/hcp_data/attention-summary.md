# RNN Model with Attention

## Handling Data
Data provided and preprocessed by the Human Connectome Project was used. It was split into training and testing sets, with the data of 100 participants selected for training and the data of the remaining 76 participants selected for evaluation. 

The train and test datasets were transformed from a dictionary with movie clip names as keys and arrays of fMRI data as values (see [summary](hcp_summary.md)) to 3-dimensional arrays of X- and Y- train and test sets. Based on the findings from Misra et al., only the first 90 seconds, equivalent to the first 90 time points, were used. 

ROI feature data was normalized with z-scores, and movie clip labels were one-hot encoded for each time step. Batches with less than 90 time steps were padded with -100.0. Thus, the X input sets had dimensions (num batches, num time steps = 90, num features = 300) and the y label sets had dimensions (num batches, num time steps = 90, num movie clips = 15).

The training datasets and testing datasets were wrapped into tensors to input into the model.

## Building the Model



## Training
The model was trained with data from 100 participants. Data was fed in batches of size 32. Cross entropy loss and the Adam optimization algorithm were used.



## Evaluating

### Permutation Testing