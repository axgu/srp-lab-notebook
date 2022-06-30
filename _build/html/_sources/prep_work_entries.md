# Prep Work
<b>Deep Learning Intro</b><br>
<i>06/01/2022, 7:45 am</i>


Synopsis: I will work through the Deep Learning Intro module.

Data: I watched the introduction video of the Deep Learning module I am following. The video covered the basics of convolutional neural networks, how they work, and visualizing them. Since the module is related to computational neuroscience, I also learned about performing analyses with fMRI data. The video also discussed comparisons between artificial and natural neural networks (animal brains) and how the two may assist each other in developments. 

I also started the first part of tutorial 1, and I created a linear deep network of depth 1 and width 200 using pytorch.

Files:
* [Deep Learning Intro Notes](./prep/deep_learning/dl_intro_notes.md)

<i>9:20 am, 100 minutes</i>

---

<b>Decoding Neural Responses Part 1</b><br>
<i>06/06/2022, 7:45 am</i>


Synopsis: I will start working through tutorial 1 of the deep learning training module.

Data: I wrote the code necessary for loading the dataset, visualizing the data, and splitting into train and test sets. I learned about the ReLU function and other non-linear activation functions. I added a ReLU layer to my deep network from last class. I also started working with the loss function and gradient descent in pyTorch.

Files:
* [Tutorial 1 Notes](./prep/deep_learning/dl_tutorial1.md)
* [Tutorial 1 Code](./prep/deep_learning/dl_tutorial1_code.ipynb)

<i>9:20 am, 100 minutes</i>

---

<b>Decoding Neural Responses Part 2</b><br>
<i>06/06/2022, 7:45 am</i>


Synopsis: I will finish working through tutorial 1 of the deep learning training module.

Data: I successfully trained my model with the parameters given in the tutorial. I also learned about neural network expressivity and the role that depth and width have on transforming data. I also reviewed the calculations that go into gradient descent/backpropagation and the difference between gradient descent and stochastic gradient descent. Finally, I learned about convolutional neural networks.

Files:
* [Tutorial 1 Notes](./prep/deep_learning/dl_tutorial1.md)
* [Tutorial 1 Code](./prep/deep_learning/dl_tutorial1_code.ipynb)

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

<b>Logistic Regression Model Permutation Test</b><br>
<i>06/23/2022, 8:39 am</i>


Synopsis: I will create a program in Python that performs a permutation test on the diabetes data with my logistic regression model.

Data: I conducted a permutation test on the 6-dimensional logistic regresison model I had previously trained. With Python, I created functions to create permutations of the dataset by shuffling the data within each feature and calculate the p-value given a distribution and value of interest. I also wrote code to generate a histogram depicting the permutation distribution in comparison to the original test value. Afterwards, I summarized my findings and conclusions.

Files:
* [Diabetes Prediction Code](./prep/log_reg/diabetes-logreg.ipynb)
* [Kaggle Dataset](https://www.kaggle.com/datasets/kandij/diabetes-dataset)
* [Diabetes Prediction Analysis](./prep/log_reg/diabetes-summary.md)
* [Bootstrap Methods and Permutation Testing Notes](./prep/bootstrap-notes.md)

<i>12:00 pm, 201 minutes</i> 

---

<b>Deep Learning Tutorials</b><br>
<i>06/24/2022, 7:30 am</i>


Synopsis: I will learn about convolutional neural networks and normative encoding models.

Data: I continued working on Deep Learning Tutorial 2 of the Neuromatch Academy material. I was having some issues running and installing PyTorch in my notebook initially. I will finish up and start Tutorial 3 this afternoon.

Files:
* [Tutorial 2 Notes](./prep/deep_learning/dl_tutorial2.md)
* [Tutorial 2 Code](./prep/deep_learning/dl_tutorial2_code.ipynb)

<i>11:00 am, 210 minutes</i>

---

<b>HCP Dataset Set Up and Software Installation</b><br>
<i>06/24/2022, 1:30 pm</i>


Synopsis: I will work on gaining access to the HCP dataset. I will also finish up the Deep Learning tutorials from Neuromatch Academy.

Data: I was able to login to the virtual machine and access the README file and the dataset itself. I also worked on installing Anaconda, though there were some difficulties with that.

Files:
* [Tutorial 2 Notes](./prep/deep_learning/dl_tutorial2.md)
* [Tutorial 2 Code](./prep/deep_learning/dl_tutorial2_code.ipynb)

<i>4:30 pm, 180 minutes</i>

---

<b>HCP Dataset</b><br>
<i>06/26/2022, 7:38 pm</i>


Synopsis: I will work on accessing and formatting the HCP data.

Data: After trying several methods, I was able to access the remote dataset and copy it onto my local device with sshfs. I will work on formatting and outputting the dataset next.

Files:
* 

<i>10:26 pm, 168 minutes</i>

---

<b>HCP Dataset Visualization</b><br>
<i>06/27/2022, 8:24 am</i>


Synopsis: I will work on formatting and outputting the dataset.

Data: I created several visuals to represent the HCP data using numpy and matplotlib. For all of my graphs, I averaged the values obtained by the participants at each (time point, ROI) coordinate, reducing the 3- or 4-dimensional array to a 2-dimensional array. 

I first made a subplot that contains 15 separate graphs, each representing a movie from the dataset. Each graph is a map, with time points on the x-axis and regions of interest (ROIs) on the y-axis; values are denoted by the color of each grid. I then visualized more specific parts of the data by creating line graphs of change in value across time points for randomly selected ROIs for the *Home Alone* movie, as well as change in value across time points for all 15 movies at a randomly selected ROI. Afterwards, I switched the time points and ROI variables and compared the *Home Alone* movie values at different ROIs for randomly selected time points. Finally, I created a video that shows the incremental change in values at each ROI for *Home Alone*. 

I also attended the LCE lab meeting virtually.

Tomorrow, I will summarize the dataset and key takeaways from visualization. I will also start working on applying logistic regression to the movie watching data.

Files:
* [Data Visualization Code](./hcp_data/data.ipynb)
* [15 Plot Colormap](./hcp_data/roi_timeseries_map.png)
* [Time Series for Selected ROIs](./hcp_data/time_rois.png)
* [Time Series for Movies](./hcp_data/time_movies.png)
* [Value Change by ROI for Selected Time Points](./hcp_data/roi_times.png)
* [Value Change by ROI Video](./hcp_data/homealone_vid.mp4)

<i>7:10 pm, 646 minutes</i>

---

<b>HCP Dataset Summary</b><br>
<i>06/28/2022, 8:13 am</i>


Synopsis: I will summarize my visualizations of the HCP dataset.

Data: I wrote a summary of the work I did yesterday with the HCP dataset, providing an overview of the data and explanations of each of my visuals. I also read the lab's paper on classifying movie clips using various machine learning models. I started working on the logistic regression model applied to the HCP data.

Files:
* [HCP Visualization Summary](./hcp_data/hcp_summary.md)
* [Learning brain dynamics for decoding and predicting individual differences by Misra et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008943)

<i>4:38 pm, 505 minutes</i>

---

<b>HCP Dataset Logistic Regression Program</b><br>
<i>06/29/2022, 8:37 am</i>


Synopsis: I will apply logistic regression to the HCP dataset.

Data: I started writing the logistic regression program in Python. More specifically, I worked on dataset pre-processing aspect. 

Files:
* [HCP Logistic Regression Code](./hcp_data/hcp_logreg.ipynb)

<i>10:00 am, 83 minutes</i>

---

<b>HCP Dataset Logistic Regression Program</b><br>
<i>06/29/2022, 12:00 pm</i>


Synopsis: I will apply logistic regression to the HCP dataset.

Data: I continued working on running logistic regression on the HCP data. I was able to convert the dictionary into a 2-dimensional array and split it into training and testing sets. I used the sklearn LogisticRegression module to assess accuracy.

Files:
* [HCP Logistic Regression Code](./hcp_data/hcp_logreg.ipynb)

<i>5:32 pm, 332 minutes</i>

---

<b>HCP Dataset Logistic Regression</b><br>
<i>06/30/2022, 8:38 pm</i>


Synopsis: I will apply logistic regression to the HCP dataset.

Data: I wrote functions to perform logistic regression with (Time point, ROI) features. I ran this model and the one provided by the sci-kit learn library several times because the classification accuracy obtained was unexpectedly high. Then, I modified the program to classify at each time point, using ROIs as features, and graphed the accruacy over time. 

Files:
* [HCP Logistic Regression Code - (Time, ROI) Features](./hcp_data/hcp_logreg_timefeature.ipynb)
* [HCP Logistic Regression Code - ROI Features](./hcp_data/hcp_logreg_indivtime.ipynb)

<i>5:40 pm, 542 minutes</i>

---