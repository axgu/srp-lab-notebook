# Prep Work
<b>Deep Learning Intro</b><br>
<i>06/01/2022, 7:45 am</i>


Synopsis: I will work through the Deep Learning Intro module.

Data: I watched the introduction video of the Deep Learning module I am following. The video covered the basics of convolutional neural networks, how they work, and visualizing them. Since the module is related to computational neuroscience, I also learned about performing analyses with fMRI data. The video also discussed comparisons between artificial and natural neural networks (animal brains) and how the two may assist each other in developments. 

I also started the first part of tutorial 1, and I created a linear deep network of depth 1 and width 200 using pytorch.

Files:
* [Deep Learning Intro Notes](./prep/dl_intro_notes.md)

<i>9:20 am, 100 minutes</i>

---

<b>Decoding Neural Responses Part 1</b><br>
<i>06/06/2022, 7:45 am</i>


Synopsis: I will start working through tutorial 1 of the deep learning training module.

Data: I wrote the code necessary for loading the dataset, visualizing the data, and splitting into train and test sets. I learned about the ReLU function and other non-linear activation functions. I added a ReLU layer to my deep network from last class. I also started working with the loss function and gradient descent in pyTorch.

Files:
* [Tutorial 1 Notes](./prep/dl_tutorial1.md)
* [Tutorial 1 Code](./prep/dl_tutorial1_code.ipynb)

<i>9:20 am, 100 minutes</i>

---

<b>Decoding Neural Responses Part 2</b><br>
<i>06/06/2022, 7:45 am</i>


Synopsis: I will finish working through tutorial 1 of the deep learning training module.

Data: I successfully trained my model with the parameters given in the tutorial. I also learned about neural network expressivity and the role that depth and width have on transforming data. I also reviewed the calculations that go into gradient descent/backpropagation and the difference between gradient descent and stochastic gradient descent. Finally, I learned about convolutional neural networks.

Files:
* [Tutorial 1 Notes](./prep/dl_tutorial1.md)
* [Tutorial 1 Code](./prep/dl_tutorial1_code.ipynb)

<i>9:20 am, 100 minutes</i>

---

<b>Jupyter Notebook Setup</b><br>
<i>06/20/2022, 9:15 am</i>


Synopsis: I will set up my Jupyter Notebook.

Data: I created and built my Jupyter Notebook with the help of the Jupyter docs. This notebook will contain both my daily log for SRP as well as all work done for the internship. I also transferred previous entries, notes, and code to my notebook.

Files:
* [Jupyter Docs](https://jupyterbook.org/en/stable/intro.html)

<i>12:34 pm, 199 minutes</i> 

---

<b>Logistic Regression</b><br>
<i>06/20/2022, 1:45 pm</i>


Synopsis: I will learn about logistic regression.

Data: I looked into logistic regression materials, including a video lecture series by Andrew Ng on YouTube and an article published by Towards Data Science. Notable aspects of study included description of a classification problem, sigmoid activation function, cost function, and gradient descent. I initially learned about logistic regression for a standard binary classification problem before extending it to multi-class classification problems. I also published my Jupyter notebook.

Files:
* [Logistic Regression](./prep/log_reg/logistic_regression.md)

<i>5:20 pm, 215 minutes</i> 

---

<b>Logistic Regression Code</b><br>
<i>06/20/2022, 8:00 pm</i>


Synopsis: I will apply logistic regression on a dataset with Python.

Data: I imported a diabetes prediction dataset from Kaggle for logistic regression. In Python, I created functions for calculating cost, gradients, and final accuracy, as well as initializing the datasets. My code also loops through and performs gradient descent a set number of times (10000) and graphs the change in cost from iteration to iteration.

Files:
* [Diabetes Prediction Code](./prep/log_reg/diabetes-logreg.ipynb)
* [Kaggle Dataset](https://www.kaggle.com/datasets/kandij/diabetes-dataset)

<i>9:10 pm, 70 minutes</i> 

---

<b>Logistic Regression Code Analysis</b><br>
<i>06/21/2022, 9:20 am</i>


Synopsis: I will analyze the results of my logistic regression diabetes prediction program.

Data: I continued working on my diabetes classification code. I created a function to split the dataset into training and testing sets to better evaluate the accuracy of the models. I also included code to graph side-by-side scatterplots of two selected features, coloring cases by observed diabetics from the dataset and predicted diabetics from the model. From this, I ran logistic regression on several combinations of two features; I selected these features since they were the most significant, as their weights from the initial model were the furthest from 0. 

Then, I tried to create a parallel coordinate plot to visualize higher dimension datasets. I initially tried to do this with just subplots in matplotlib, but it was difficult to set the xtick labels correctly and have the legend show. This method was also relatively slow. As a result, I decided to convert my numpy arrays into a pandas dataframe and use the pandas method plotting.parallel_coordinates(). I was able to obtain a cleaner visual from this method, although the shared normalized y-axis scale makes it difficult to immediately notice differences in most features between subjects with diabetes and those without. However, the figure still looks very cluttered.

Files:
* [Diabetes Prediction Code](./prep/log_reg/diabetes-logreg.ipynb)
* [Kaggle Dataset](https://www.kaggle.com/datasets/kandij/diabetes-dataset)
* [Diabetes Prediction Analysis](./prep/log_reg/diabetes-summary.md)

<i>5:00 pm, 460 minutes</i> 

---

<b>Logistic Regression Code Analysis and Bootstrap Methods and Permutation Testing</b><br>
<i>06/22/2022, 8:28 am</i>


Synopsis: I will continue working on my analysis of the results of my logistic regression diabetes prediction program.

Data: I edited my analysis of the diabetes prediction program. I mainly worked on including more information about each of the models I tested. This involved listing inputs and outputs clearly and labeling provided figures/graphs. I also did some editing to clarify parts of my summary that were too vague or confusing.

Afterwards, I read about and took notes on bootstrap and permutation testing as methods of computational statistical inference. More specifically, the chapter included details about bootstrap distributions, resampling, accuracy, bootstrap confidence intervals, and permutation (significance) testing.

Files:
* [Diabetes Prediction Code](./prep/log_reg/diabetes-logreg.ipynb)
* [Kaggle Dataset](https://www.kaggle.com/datasets/kandij/diabetes-dataset)
* [Diabetes Prediction Analysis](./prep/log_reg/diabetes-summary.md)
* [Bootstrap Methods and Permutation Testing Notes](./prep/bootstrap-notes.md)

<i>3:23 pm, 415 minutes</i> 

---