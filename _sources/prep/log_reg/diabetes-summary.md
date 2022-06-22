# Diabetes Prediction Summary

## Initial Models
Logistic regression was performed for a binary classification problem to predict whether a subject is diabetic or non-diabetic. Input variables provided in the dataset included:
* Number of pregnancies (int)
* Glucose level (int)
* Blood pressure (int)
* Skin thickness (int)
* Insulin (int)
* BMI (float)
* Diabetes pedigree function (float)
* Age (int)

Outcomes were denoted by a 1 for diabetic and a 0 for non-diabetic.

Functions were created in Python to initialize the data, split the dataset into training and testing sets, normalize input values, add a bias, calculate cost, gradients, and model accuracy, iterate through gradient descent, and visualize the data. Initially, the model was trained on all provided features, resulting in the following output:

### 8-Dimensional Model

Input variables:
* Number of pregnancies (int)
* Glucose level (int)
* Blood pressure (int)
* Skin thickness (int)
* Insulin (int)
* BMI (float)
* Diabetes pedigree function (float)
* Age (int)

Output:
* Final weights:&ensp;[-0.66405417 &ensp; 0.30734772 &ensp; 0.71401776 &ensp;-0.11287652 &ensp; -0.00607035 &ensp; 0.03314956 &ensp; 0.52259872 &ensp; 0.2591092 &ensp; 0.16235891]
* Final cost:&ensp;-0.0017505502766420476
* Training Accuracy:&ensp;0.7673611111111112
* Testing Accuracy:&ensp;0.78125


Since all input values were normalized, the final weights furthest from 0 reflect the features that have the largest impact on the classification. Conversely, the classes corresponding to the weights very close to 0 are less significant. Thus, skin thickness and insulin levels, with weights -0.0060735 and 0.03314956, were considered least impactful.

---

Another logistic regression model was trained and tested on the remaining 6 classes (i.e. excluding skin thickness and insulin levels). The resulting classification accuracies for training and testing sets were similar to the initial 8-dimensional model: 

### 6-Dimensional Model

Input variables:
* Number of pregnancies (int)
* Glucose level (int)
* Blood pressure (int)
* BMI (float)
* Diabetes pedigree function (float)
* Age (int)

Output:
* Final weights:&ensp; [-0.89146127 &ensp; 0.42844742 &ensp; 0.9595772 &ensp; -0.22267593 &ensp; 0.75103708 &ensp; 0.31908514 &ensp; 0.10940684]
* Final cost:&ensp; -0.050473260107638356
* Training Accuracy:&ensp; 0.7743055555555556
* Testing Accuracy:&ensp; 0.7760416666666666


As indicated by the output results of this model as well as the previous model, glucose is most impactful on diabetes prediction, as its corresponding weight is furthest from 0. This observation is also corroborated by the parallel coordinate plot of the test data:

```{image} ./parallel-plot.png
:name: label
```

The glucose column has the most distinct separation between diabetic and non-diabetic cases.

---

## Glucose Comparison

Since both previous applications of logistic regression on the diabetes dataset indicated that glucose was the most significant variable, models were created to perform logistic regression on glucose and another variable that also had a greater impact. Disregarding the bias, as this is the same for all cases, the features with the most extreme weights after glucose, from most significant to least signficiant, were BMI, pregnancies, and diabetes pedigree function. Performing logistic regression on these variables and glucose yielded similar test accuracies of 76.0417%, 77.0834%, and 77.6042%, respectively. This can also be seen in the side-by-side scatterplots below comparing predicted diabetics to observed diabetics for each of the three models.

### 2-Dimensional Models Comparison

|                         |                           |                            |                           |
| :---------------------- | :------------------------ |  :------------------------ | :------------------------ |
|    **Input features**   | Glucose level (int)<br>BMI (float) | Glucose level (int)<br>Number of pregnancies (int) | Glucose level (int)<br>Diabetes pedigree function (float) |
|     **Final weights**   | [-0.84826529  1.02577726  0.68402623] | [-0.80287488  1.08189954  0.38894197] | [-0.63104608  0.84221055  0.27509572] |
|      **Final cost**     | -0.058537166596216314 | -0.065769539953255 | -0.02408674110918849 |
|  **Training Accuracy**  | 0.7586805555555556 | 0.7413194444444444 | 0.7465277777777778 |
|    **Test Accuracy**    |  0.7604166666666666 | 0.7708333333333334 | 0.7760416666666666 |

![](./glucose-bmi.png) ![](./glucose-pregnancies.png) ![](./glucose-dpf.png)


### Decision Boundary

The black line in each of the "Predicted Diabetes" scatterplots shows the decision boundary. This was calculated by setting the sigmoid function $h_{\theta}(x) = 0.5$. Thus, the boundary is determined by when $\theta_{0} + \theta_{1}\cdot x_{1} + \theta_{2}\cdot x_{2} = 0$.

Solving for $x_{2}$, the y-axis variable of the scatterplot, gives

$$
x_{2} = \frac{-\theta_{0} - \theta_{1}\cdot x_{1}}{\theta_{2}}
$$

This is a linear relationship between the x-axis variable, $x_{1}$, and the y-axis variable, $x_{2}$.

---

Finally, logistic regression was performed to determine the effect of removing glucose from the selected features. A model was created, trained, and tested on BMI and pregnancy values. The result is a lower classification accuracy rate, as observed from the output:

### 2-Dimensional Model Without Glucose

Input variables:
* Number of pregnancies (int)
* BMI (float)

Output:
* Final weights:&ensp; [-0.62254392 &ensp; 0.61786381 &ensp; 0.4043365]
* Final cost:&ensp; -0.028591104587616968
* Training Accuracy:&ensp; 0.6996527777777778
* Test Accuracy:&ensp; 0.671875

```{image} ./bmi-pregnancies.png
:name: label
```


## Resources
* [Diabetes Prediction Code](./diabetes-logreg.ipynb)
* [Kaggle Dataset](https://www.kaggle.com/datasets/kandij/diabetes-dataset)
* [Parallel Plot Graph by Yan Holtz](https://www.python-graph-gallery.com/150-parallel-plot-with-pandas)
* [Matplotlib Docs](https://matplotlib.org/stable/api/axes_api.html)