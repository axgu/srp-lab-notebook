# Feedforward Neural Network

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
