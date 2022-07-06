# HCP Data

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

<b>HCP Dataset Logistic Regression</b><br>
<i>07/01/2022, 7:30 am</i>


Synopsis: I will apply logistic regression to the HCP dataset and summarize the results.

Data:  

Files:
* [HCP Logistic Regression Overview](./hcp_data/hcp_logreg.md)
* [HCP Logistic Regression Code - ROI Features](./hcp_data/hcp_logreg_indivtime.ipynb)

<i>4:30pm, 540 minutes</i>

---

<b>HCP Dataset Logistic Regression Summary</b><br>
<i>07/05/2022, 10:00 am</i>


Synopsis: I will apply logistic regression to the HCP dataset and summarize the results.

Data: I was able to resolve my confusion about the logistic regression model. I modified my previous logistic regression ROI feature program so that the model is trained once with data across all 90 time points considered. Then, the test data was split by time points. The fitted model was evaluated with this data and classification accuracy was calculated at each time point to create a time series plot.

Files:
* [HCP Logistic Regression Overview](./hcp_data/hcp_logreg.md)
* [HCP Logistic Regression Code - ROI Features](./hcp_data/hcp_logreg_indivtime.ipynb)

<i>6:17 pm, 497 minutes</i>

---

<b>Permutation Testing</b><br>
<i>07/06/2022, 8:37 am</i>


Synopsis: I will use a permutation test to evaluate the effectiveness of the logistic regression model.

Data:  

Files:
* [HCP Logistic Regression Overview](./hcp_data/hcp_logreg.md)
* [Permutation Testing](./hcp_data/hcp_log_reg_permtest.ipynb)

<i></i>

---