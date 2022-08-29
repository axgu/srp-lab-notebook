# Seq2Seq Attention Model

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