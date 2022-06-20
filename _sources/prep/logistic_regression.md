# Logistic Regression
* Models probabilities for classification
    * Values strictly between 0 (negatife class) and 1 (positive class)
    * Linear regression not ideal because:
        * Values can be > 1 or < 0
        * Extreme values influence linear regression model
* Used when dependent variable is categorical
    1. Binary logistic regression - categorical response has only two possible outcomes
    2. Multinomial logistic regression - three or more categories without ordering
    3. Ordinal logistic regression - three or more categories with ordering

### Machine Learning Process
1. Identify features and labels (data)
2. Set initial weights for model
3. Loop until model is close
    1. Calculate cost
    2. Calculate gradient
    3. If length of gradient vector is close to 0, stop; otherwise, adjust weights based on the gradient

### Hypothesis Representation
* Classifier should output between 0 and 1 only
* Sigmoid function:

  $$
  g(z) = \frac{1}{1 + \exp(-z)}
  $$

* Hypothesis representation: 

$$
h_{\theta}(x) = \frac{1}{1 + \exp(-\theta^{T}x)}
$$

* $h_{\theta}(x)$ = estimated probability that y = 1 on input x
* $h_{\theta}(x) = P(y=1|x; \theta)$
* Since y must equal 0 or 1, $P(y=0|x; \theta) = 1 - P(y=1|x; \theta)$

### Decision Boundary
* Separates region where the hypothesis predicts y = 1 from the region where the hypothesis predicts y = 0
* Property of the hypothesis, not the training set
* Higher order polynomials result in more complex decision boundaries

### Cost Function
* Gradient descent will converge into global minimum only if the function is convex

$$
J_{\theta} = \frac{1}{m}\sum_{i=1} ^{m} \text{Cost}(h_{\theta}(x^{(i)}), y^{(i)})

\text{Cost}(h_{\theta}(x), y) = \begin{cases}
-\log(h_{\theta}(x)),  & \text{if $y = 1$} \\[2ex] 
-\log(1 - h_{\theta}(x)),  & \text{if $y = 0$}
\end{cases}
$$

For the y = 1 case:
* Cost = 0 if $y = 1$, $h_{\theta}(x) = 1$
* As $h_{\theta}(x)\to0$, $Cost\to\infty$

For the y = 0 case:
* Cost = 0 if $y = 0$, $h_{\theta}(x) = 0$
* As $h_{\theta}(x)\to1$, $Cost\to\infty$

If there is a large difference between $h_{\theta}(x)$ and $y$, the learning algorithm is penalized by a very large cost.

#### Simplified Cost Function
We can rewrite the cost function as:

$$
\text{Cost}(h_{\theta}(x), y) = -y\log(h_{\theta}(x)) - (1-y)\log(1-h_{\theta}(x))
$$

Checking our cases will result in our first definition:
* If $y = 1$: $\text{Cost}(h_{\theta}(x), y) = -1(\log(h_{\theta}(x))) - 0(\log(1-h_{\theta}(x))) = -\log(h_{\theta}(x))$
* If $y = 0$: $\text{Cost}(h_{\theta}(x), y) = -0(\log(h_{\theta}(x))) - 1(\log(1-h_{\theta}(x))) = -\log(1 - h_{\theta}(x))$

So our final logistic regression cost function is:

$$
J_{\theta} = -\frac{1}{m}\sum_{i=1} ^{m} y^{(i)}\log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}\log(1 - h_{\theta}(x^{(i)})))
$$

### Gradient Descent
* We use gradient descent to minimize our cost function
* To do this, we repeatedly update each parameter:

$$
\theta_{j} := \theta_{j} - \alpha\frac{\partial}{\partial \theta_{j}}J(\theta)
\theta_{j} := \theta_{j} - \alpha\sum_{i=1} ^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_{j} ^{(i)}
$$

### Advanced Optimization
* Optimization algorithms:
    * Conjugate gradient
    * BFGS
    * L-BFGS
* Advantages:
    * No need to manually pick $\alpha$
    * Often faster than gradient descent
* Disadvantages:
    * More complex

### Multiclass Classification: One-vs-all
* Turn the problem into several binary classification problems

$$
h_{\theta} ^{(i)}(x) = P(y=i|x; \theta) \quad (i = 1,2,3)
$$

* Train a logistic regression classifier $h_{\theta} ^{(i)}(x)$ for each class/category $i$ to predict the probability that $y = i$
* To make a prediction on a new input $x$, calculate and pick the class $i$ that maximizes $h_{\theta} ^{(i)}(x)$

## Resources
* [Logistic Regression Lecture by Andrew Ng](https://www.youtube.com/watch?v=-EIfb6vFJzc)
* [Towards Data Science Logistic Regression Overview](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
* [Interpretable ML Book by Christoph Molnar](https://christophm.github.io/interpretable-ml-book/logistic.html)