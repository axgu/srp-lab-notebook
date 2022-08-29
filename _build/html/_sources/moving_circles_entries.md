# Moving Circles Dataset

**Moving Circles RNN**

*08/18/2022, 8:40 am*


Synopsis: I will work with a moving circles dataset.

Data: I trained an LSTM model, similar in structure to the one used on the movie watching dataset, on the movie circles data. I also evaluated binary classification accuracy with the test dataset, and tried "breaking" the model using random inputs. 

Files:
* [Moving Circles LSTM](./appr_retr/appr_retr_rnn.ipynb)

*3:25 pm, 405 minutes*

---

**Moving Circles Transformer**

*08/19/2022, 8:45 am*


Synopsis: I will work with a moving circles dataset.

Data: I applied a transformer encoder architecture similar to the one used for multiclass classification for the movie clip dataset on the emoprox dataset. From training and evaluating performance of this neural network, I found that accuracy was periodic.

Files:
* [Moving Circles Transformer](./appr_retr/appr_retr_transformer.ipynb)

*3:00 pm, 315 minutes*

---

**Moving Circles Dataset**

*08/22/2022, 9:15 am*

Synopsis: I will learn about the moving circles dataset.

Data: I got a better understanding of the moving circles dataset and what the label dataset represents. I also finished running a random inputs test on the transformer model.

Files:
* [Emoprox Dataset Video](https://pessoalab.slack.com/files/UC26JSETT/F037ZN8NQ9F/movie_subj013__2_.mp4)
* [Moving Circles Transformer](./appr_retr/appr_retr_transformer.ipynb)

*3:00 pm, 275 minutes*

---

**Padding Labels**

*08/23/2022, 8:55 am*

Synopsis: I will work on padding the y set for the moving circles dataset.

Data: I worked on introducing a new class for padded values in the y-train and y-test set, as well as vectorizing these label datasets. To ensure the model did not prioritize the padding class, I calculated and passed weights to the loss function. I then retrained and tested the LSTM and transformer models.

Files:
* [Moving Circles LSTM](./appr_retr/appr_retr_rnn.ipynb)
* [Moving Circles Transformer](./appr_retr/appr_retr_transformer.ipynb)

*4:30 pm, 375 minutes*

---

**Padding and Masking**

*08/26/2022, 8:30 am*

Synopsis: I will work on a new padding method for the y set for the moving circles dataset.

Data: The results from the previous padding method indicated that the models were still learning the padding class as a label and assigning it to some input sequences. So, I modified the one-hot encoding so that there were only 2 classes, but padded values were assigned a [0,0] vector and left out of accuracy calculations. However, I found from running both models that while classification accuracy improved, they varied periodically across time.

I also attempted to add padding masks to the transformer architecture. However, I kept receiving nan loss values.

Files:
* [Moving Circles LSTM](./appr_retr/appr_retr_rnn.ipynb)
* [Moving Circles Transformer](./appr_retr/appr_retr_transformer.ipynb)

*4:50 pm, 420 minutes*

---