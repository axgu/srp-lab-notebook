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
* [LSTM PyTorch Model (.py)](./hcp_data/lstm.py)

*2:33 pm, 318 minutes*

---

**Attention Methods**

*07/18/2022, 8:50 am*


Synopsis: I will learn about attention methods.

Data: I learned about sequence to sequence models, specifically about encoder decoder architectures. I also looked into LSTM models to understand how they worked. Then, I learned about the bottleneck problem in standard encoder decoder models and the solution provided by attention. I looked into several attention mechanisms within seq2seq problems, as well as a basic overview of self-attention.

Files:
* [Attention Methods Notes](./hcp_data/attention_methods.md)

*5:50 pm, 540 minutes*

---

**Attention Model**

*07/19/2022, 9:00 am*


Synopsis: I will implement an attention model.

Data: I worked on adding figures to my attention methods notes. Then, I started working in Python with PyTorch to build an encoder class and a decoder class.

Files:
* [Attention Methods Notes](./hcp_data/attention_methods.md)
* [Attention Model (.py)](./hcp_data/attention.py)

*5:00 pm, 480 minutes*

---

**Attention Model**

*07/20/2022, 9:00 am*


Synopsis: I will implement an attention model.

Data: I finished building the encoder, decoder, and attention classes in PyTorch. Then, I worked on preprocessing the dataset into training/testing and input/output data. I also created functions for padding the arrays and vectorizing the movie clip labels. I wrote code to train the model that incorporates the attention mechanism. Finally, I looked into creating masks with PyTorch, and will implement that, as well as model evaluation, tomorrow.

Files:
* [Attention Model (.py)](./hcp_data/attention.py)
* [Data Prep (.py)](./hcp_data/lstm_data_prep.py)

*5:33 pm, 513 minutes*

---

**Attention Model**

*07/21/2022, 8:00 am*


Synopsis: I will continue working on my attention model.

Data: I fixed errors within my encoder and decoder classes. I also built an LSTM  model in PyTorch, as well as train and test methods. An issue with my LSTM model when I try to train it is that the loss stays constant over all epochs.

Files:
* [Attention Model (.py)](./hcp_data/attention.py)
* [Model Evaluation (.py)](./hcp_data/eval_model.py)

*7:30 pm, 690 minutes*

---

**Attention Model Masking**

*07/25/2022, 9:00 am*


Synopsis: I will continue working on my attention model.

Data: I worked on adding padding and masking to the data in my LSTM and encoder-decoder attention model. I was also able to fix the issue with my LSTM model where the training loss was not decreasing by transposing the output from the LSTM and the actual y arrays from (batch size x sequence length x class) to (batch size x class x sequence length). Then I continued training the models.

Files:
* [Attention Model (.py)](./hcp_data/attention.py)

*6:15 pm, 555 minutes*

---

**Attention Model Training**

*07/26/2022, 8:30 am*


Synopsis: I will continue working on my attention model.

Data: I continued training my LSTM and Seq2Seq models, experimenting with learning rate and epoch size. I also saved them to my local disk and plotted the resulting accuracies for comparison.

Files:
* [Attention Model (.py)](.hcp_data/attention.py)
* [LSTM Model (.py)](.hcp_data/lstm.py)

*6:25 pm, 595 minutes*

---

**Attention Model Permutation Test**

*07/27/2022, 9:25 am*


Synopsis: I will perform a permutation test on my LSTM and Seq2Seq attention model.

Data: I wrote functions to shuffle each feature along the time steps in a batch. I then ran a permutation test with 100 resamples and plotted a comparison of accuracy between logistic regression, LSTM, and attention performance.

Files:
* [RNN Permutation Test Code (.py)](./hcp_data/rnn_perm_test.py)
* [Attention Model](./hcp_data/attention-book.ipynb)
* [LSTM Model](./hcp_data/lstm-book.ipynb)
* [Visual Model Comparison](./hcp_data/hcp_plot.ipynb)

*5:20 pm, 475 minutes*

---

**Attention Model Analysis**

*07/28/2022, 8:30 am*


Synopsis: I will summarize what I have implemented.

Data: The permutation test for the attention Seq2Seq model indicated around 80% accuracy, which is unreasonably high. So, I worked on trying to "break" my model. Each method I utilized was performed 3 times. I tested by randomizing labels for each (batch, time step), which yielded an 8% accuracy for all time steps, and randomizing labels for each batch but remaining the same for all time steps, which yielded a 22% accuracy. I then tried re-running the permutation test (i.e. shuffling the data across time steps for each feature). Afterward, I tried randomly generating input feature data from a normal distribution. Both of these evaluations also yielded approximately an 80% accuracy for all time steps.

Files:
* [RNN Permutation Test Code (.py)](./hcp_data/rnn_perm_test.py)
* [Attention Model](./hcp_data/attention-book.ipynb)

*7:02 pm, 632 minutes*

---

**Attention Model Summary**

*07/29/2022, 8:39 am*


Synopsis: I will continue analyzing why my attention model has such a high performance with permuted samples.

Data: I met with my mentors briefly to discuss possible issues with my attention model. I also tried to determine the issue with my attention encoder decoder architecture.

Files:
* [RNN Permutation Test Code (.py)](./hcp_data/rnn_perm_test.py)
* [Attention Model](./hcp_data/attention-book.ipynb)

*4:00 pm, 441 minutes*

---

**Neural Networks**

*08/01/2022, 8:27 am*


Synopsis: I will continue analyzing why my attention model has such a high performance with random inputs.

Data: I learned more about attention mechanism through a meeting with mentors. I also watched videos and read about how neural networks work and backpropagation.

*6:00 pm, 573 minutes*

---

**Feedforward Network**

*08/02/2022, 8:53 am*


Synopsis: I will build a feedforward neural network.

Data: I built, trained, and evaluated a feedforward neural network in PyTorch that considers time steps blindly. I also tried adding the attention mechanism discussed in the meeting yesterday to a recurrent neural network and trained the model.

Files:
* [RNN Permutation Test Code (.py)](./hcp_data/rnn_perm_test.py)
* [Attention Model](./hcp_data/attention-book.ipynb)

*6:20 pm, 567 minutes*

---

**Feedforward with Attention**

*08/03/2022, 8:38 am*


Synopsis: I will summarize what I did with the feedforward neural network.

Data: I wrote a summary of my work with the feedforward neural network. I provided details about handling the dataset, architecture specification, training, and evaluating model performance. I had another meeting with my mentors to discuss adding an attention layer to a feedforward network. Then I tried implementing what was discussed. However, when I tried training the model, every epoch took over an hour and a half and the loss for several iterations was outputted as infinity.

Files:
* [Feedforward Network Summary](./hcp_data/ff_summary.md)
* [Feedforward Attention](./hcp_data/ff_attention.ipynb)

*5:30 pm, 532 minutes*

---

**Feedforward with Attention**

*08/04/2022, 8:48 am*


Synopsis: I will continue training the feedforward/attention architecture and evaluate its performance.

Data: I retrained the feedforward with attention model since my computer restarted and training was incomplete. I also modified the feedforward network so that it trains data from each time point separately as its own classification problem. While my model was training, I also worked on organizing my code and Jupyter notebook.

Files:
* [Feedforward Network](./hcp_data/ff.ipynb)
* [Feedforward Attention](./hcp_data/ff_attention.ipynb)

*5:30 pm, 532 minutes*

---

**Feedforward with Attention Summary**

*08/05/2022, 9:00 am*


Synopsis: I will continue training the feedforward/attention architecture and evaluate its performance.

Data: I summarized work with the feedforward attention architecture, including documenting the math for the attention mechanism and summarizing parts of the code written. 

Files:
* [Feedforward Attention Summary](./hcp_data/ff_sattention_summary.md)
* [Feedforward Attention](./hcp_data/ff_attention.ipynb)

*3:30 pm, 390 minutes*

---

**Feedforward Wrap-Up**

*08/08/2022, 8:40 am*


Synopsis: I will wrap up work with the feedforward network and feedforward attention model.

Data: I ran permutation tests on the model fitted at each time point in the feedforward neural network. I also wrote another program to perform logistic regression in a similar manner, by considering data from each time step as a separate classification problem. This model was trained and tested on HCP data and fed random feature values. Then, I retrained the feedforward attention model at a smaller learning rate with more epochs. I also sent Ms. Bosse and email updating my progress.

Files:
* [Feedforward Neural Network](./hcp_data/ff.ipynb)
* [Logistic Regression](./hcp_data/logistic_regression.ipynb)
* [Feedforward Attention](./hcp_data/ff_attention.ipynb)

*3:40 pm, 420 minutes*

---
