# HCP Logistic Regression
## Time Series Classification with ROI Features
Multinomial logistic regression was applied to the HCP dataset for 15-way time series classification of movie clips. Code can be found [here](hcp_logreg_indivtime.ipynb).

### Dataset Creation
The dataset was transformed from a dictionary with movie clip names as keys and arrays of fMRI data as values (see [summary](hcp_summary.md)) to 2-dimensional arrays of X- and Y- train and test sets at each time point considered. Based on the findings from Misra et al., only the first 90 seconds, equivalent to the first 90 time points, were used. 

At each time point, a general 2-dimensional array was created. Each row was designated for a different subject-movie clip combination. For 'testretest' cases, different runs from the same subject were also compiled as separate rows. The number of rows varied between time points, as both 'overcome' and 'testretest' are clips shorter than 90 seconds. Columns were designated for distinct ROIs, as well as the movie label and participant subject, which were used for splitting up the array. Thus, each array had 300 feature columns, in addition to a movie label column and a participant number column.

The data was split with 100 participants for training and 76 participants for testing. A list of participants used for the testing data set was randomly generated. Participant numbers within this list had corresponding ROI feature data stored in X_test and corresponding movie labels stored in y_test. The same was done for participants not selected in the test set, except with X_train and y_train, respectively. All feature data in both training and testing sets was normalized using z-scores.

### Logistic Regression Model
At each of the first 90 time points, a multinomial logistic regression model was created and fitted to the training data, using the sci-kit learn library in Python. Max_iters was set to 1000 and all other default parameter values were used. The model was then evaluated with the testing data, and the accuracy was saved. Classification accuracy at each time point was plotted, as shown below:

![](../_build/jupyter_execute/hcp_data/hcp_logreg_indivtime_6_0.png)


---


## Classification with (Time point, ROI) Features
A multinomial logistic regression model was applied to the HCP dataset for 15-way classification of movie clips, using features with both a time and ROI aspect. Code can be found [here](hcp_logreg_timefeature.ipynb).

### Dataset Creation
The dataset was transformed from a dictionary with movie clip names as keys and arrays of fMRI data as values (see [summary](hcp_summary.md)) to 2-dimensional arrays of X- and Y- train and test sets. Since 'overcome' was 65 seconds long, only the first 65 time points were considered.

An overall 2-dimensional array was created prior to splitting into X/Y train/test sets. Each row represented data for a subject-movie clip combination. Similar to the previous time series classification method, for 'testretest' cases, different runs from the same subject were also compiled as separate rows. Columns in the array contained time point-ROI combinations, as well as the movie label and participant subject; there were 65x300 feature columns, in addition to a movie label column and a participant number column.

The data was split with 100 participants for training and 76 participants for testing. Similar to the method above, a list of participants used for the testing data set was randomly generated. Participant numbers within this list had corresponding ROI feature data stored in X_test and corresponding movie labels stored in y_test. The same was done for participants not selected in the test set, except with X_train and y_train, respectively. All feature data in both training and testing sets was normalized using z-scores.

### Logistic Regression Model
A multinomial logistic regression model was created and fitted to the training data, using the sci-kit learn library in Python. Max_iters was set to 1000 and all other default parameter values were used. The model was then evaluated with the testing data, and the accuracy found was around 96%.