# Recurrent Neural Network

**Recurrent Neural Networks**

*07/11/2022, 8:40 am*


Synopsis: I will learn about LSTM models and apply one to the HCP dataset.

Data: I sent Ms. Bosse an email updating her with my progress for the first few weeks of my internship. I also took notes on recurrent neural networks, and more specifically, Gated Recurrent Units. I also read about LSTMs, another RNN architecture. Finally, I looked into implementing an RNN using different Python libraries, including Pytorch, Tensorflow, and sklearn.

Files:
* [Recurrent Neural Network Notes](./recurrent_nn.md)

*5:12 pm, 512 minutes*

---

**RNN Data Processing**

*07/12/2022, 9:02 am*


Synopsis: I will apply a GRU network to the HCP dataset.

Data: I learned about specific parameters and requirements with implementing a GRU model using Tensorflow and Pytorch. I also wrote code to split the HCP dataset into training and testing dictionaries. Then, I wrote a function that reshaped these dictionaries into 3D X (input) arrays of shape (batch x time x features) and 2D y (output) arrays of shape (batch x time). Instead of using all possible time points, I limited the sequence length to the first 90 time points. However, because some movie clips were sampled for times shorter than 90 seconds, I learned about padding and masking the data.

Files:
* [RNN Data Prep Program](./hcp_data/rnn_data_prep.ipynb)

*5:26 pm, 504 minutes*

---

**RNN Training**

*07/13/2022, 9:00 am*


Synopsis: I will apply a GRU network to the HCP dataset.

Data: I built a GRU model using Tensorflow/Keras with a Masking layer, GRU layer, and TimeDistributed/Dense layer. The final layer was used to allow separate time calculations and also apply the softmax activation function to the previous outputs, since labels were one-hot encoded into 15D vectors. The model was trained with X and y training data, batch_size = 32, epoch = 50, and validation_split = 0.2. It was then used to predict clips of the input test data. Accuracy functions were written to evaluate the model, though the results indicate that there were some issues with masking, since there is a noticeable decrease in performance for time points after 65 seconds and 84 seconds, which is when overcome and testretest clips end, respectively. 

Files:
* [RNN Data Prep Program](./hcp_data/rnn_data_prep.ipynb)

*5:01 pm, 481 minutes*

---

**RNN Masking and Evaluation**

*07/14/2022, 8:31 am*


Synopsis: I will learn more about masking and padding data and evaluate the performance of the neural network.

Data: I learned about masking and padding in Keras, and tried to apply it to my model. An issue I encountered was that the Keras frameworks skip over the masked time step for all samples, not just the padded ones. I also corrected my accuracy function. Finally, I worked on optimizing the performance of the LSTM by tuning hyperparameters, such as changing batch size, epochs, dropout rate, etc.

Files:
* [RNN Data Prep Program](./hcp_data/rnn_data_prep.ipynb)

*4:40 pm, 489 minutes*

---

**RNN with PyTorch**

*07/15/2022, 9:15 am*


Synopsis: I will work on fixing the masking issue of the neural network.

Data: I learned about implementing neural networks with PyTorch instead of Tensorflow.

Files:
* [LSTM PyTorch Model](./hcp_data/lstm.py)

*2:33 pm, 318 minutes*

---

**Attention Methods**

*07/14/2022, 8:50 am*


Synopsis: I will learn about attention methods.

Data: I learned about sequence to sequence models, specifically about encoder decoder architectures. I also looked into LSTM models to understand how they worked. Then, I learned about the bottleneck problem in standard encoder decoder models and the solution provided by attention. I looked into several attention mechanisms within seq2seq problems, as well as a basic overview of self-attention.

Files:
* [Attention Methods Notes](./hcp_data/attention_methods.md)

*5:50 pm, 540 minutes*

---