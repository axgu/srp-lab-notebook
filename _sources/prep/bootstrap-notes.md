# Bootstrap Methods and Permutation Testing Notes
## The Bootstrap Idea
* Computational method to conduct statistical inference
* Can be applied when theory fails (i.e. conditions for traditional inference not all met): randomization/lack of bias necessary, but large enough sample size and normal population distribution not needed
* Approximate sampling distribution from one sample:
    1. **Resampling** - repeatedly sample **with** replacement from initial random sample
    2. **Bootstrap distribution** - based on many resamples; represents the sampling distribution of the statistic, based on many samples
        * **Shape** - Approximately the same as the sampling distribution
        * **Center** - Centered at original statistic value instead of parameter value
        * **Spread** - Approximately the same as the sampling distribution
* Bootstrap distribution of resample means only used to estimate how the sample mean of _one_ sample of size n would vary due to random sampling
* Apply plug-in principle: substitute sample for population

Suppose $B$ resamples are taken. Let the means of these resamples be $\overline{x}^{*}$. Then

$$
\text{mean}_{\text{boot}} = \frac{1}{B} \sum \overline{x}^{*}


\text{SE}_{\text{boot}} = \sqrt{\frac{1}{B - 1} \sum (\overline{x}^{*} - \text{mean}_{\text{boot}})^{2}}
$$

* **Bootstrap standard error** $\text{SE}_{\text{boot}}$ - standard deviation of the bootstrap distribution of the statistic


## First Steps in Using the Bootstrap
* **Shape** - Approximates the shape of the sampling distribution, so the bootstrap distribution can be used to check normality of the sampling distribution
* **Center** - Compare whether bootstrap distribution of the statistic is centered at original statistic value
    * **Bias** - Difference between mean of a statistic's sampling distribution and the true value
    * **Bootstrap estimate of bias** - Difference between mean of bootstrap distribution and original sample statistic value
* **Spread** - Bootstrap standard error of a statistic is the standard deviation of its bootstrap distribution, which is an estimate of the standard deviation of the sampling distribution

### Bootstrap t confidence intervals
* Get a confidence interval for the parameter by using the bootstrap standard error and _t_ distribution if:
    * Bootstrap distribution shows a normal shape
    *Bootstrap distribution has a small bias

* **Trimmed mean** - mean of only the center observations in a data set. _Ex: 25% trimmed mean ignores smallest 25% and largest 25% of observations._

* An approximate level _C_ confidence interval for the parameter that corresponds to the statistic is 

$$
\text{statistic} \pm t^{*}\text{SE}_{\text{boot}}
$$

* $t^{*}$ is the critical value corresponding to _C_ of the $t(n-1)$ distribution

### Bootstrap to compare two groups

Given independent SRSs of sizes $n$ and $m$ from _two_separate populations:
1. Draw resamples of sizes $n$ and $m$ from their respective initial samples. Compute a statistic to compare the two groups.
2. Repeat
3. Construct bootstrap distribution


## Bootstrap Distribution Accuracy
* Sources of variation among bootstrap distributions:
    1. Choosing a random original sample from the population
    2. Resampling from the original sample
* Bootstrap distributions inform about shape, bias, and spread of sampling distribution
    * Do not have the same center as sampling distribution
* Larger samples preferred because bootstrap distributions from small samples do not as closely mimic shape and spread of sampling distribution
* Resampling with 1000+ resamples introduced very little additional variation
* Bootstrap inference based on smaller initial samples is unreliable for medians and quartiles since those are calculated from just a few of the sample observations
    * The initial sample excludes influential values in the population


## Bootstrap Confidence Intervals
### Bootstrap percentile confidence intervals
* Interval between 2.5% and 97.5% percentiles of bootstrap distribution of a statistic is a 95% bootstrap percentile confidence interval
* Estimate of bias must be small
* Percentiles do not ignore skewness; not symmetric
* Compare with t interval - if they do not agree, do not use either


* **Accuracy** - Produces intervals that capture parameter _C_% of the time; other $1-C$% misses equally between high and low
* Generally, the t and percentile intervals may NOT be sufficiently accurate when:
    * statistic is strongly biased
    * sampling distribution is skewed
    * high accuracy is necessary because of high stakes


### BCa and tilting
* WARNING: use cautiously when sample sizes are small
* **Bootstrap bias-corrected accelerated (BCa) interval** - modification of percentile method that adjusts percentiles to correct for bias and skewness
    * Requires more than 1000 resamples for high accuracy; 5000 or more for very high accuracy
* **Bootstrap tilting interval** - adjusts process of random resampling 


## Significance Testing Using Permutation Tests
* Significance tests - explain whether an observed results could reasonably occur "by chance" from random sampling
    1. Choose statistic that measures effect
    2. Construct sampling distribution the statistic would have without the effect
    3. Locate observed statistic and decide based on location (main part, tail, etc.) whether the effect exists

    * **Null hypothesis** - Statement that effect is not present
    * **P-value** - probability of observed result or something more extreme occurring "by chance" (i.e. without effect)

* Resampling for significance tests must be consistent with the null hypothesis to estimate sampling distribution when null hypothesis is true
* **Permutation tests** - tests based on resampling
    * Permutation resample - drawn **without** replacement
    * Permutation distribution
* Permutation tests give accurate P-values when sampling distribution is skewed
* Resampling methods that move observations between two groups require that the two populations are identical (in shape, center, spread) when null is true
* Sources of variability
    * Original sample chosen at random from population
    * Resamples chosen at random from sample
        * Added variation due to resampling is generally small

* If the true one-sided P-value is $p$, the standard deviation of the estimated _P_-value is $\sqrt{p(1-p) / B}$
    * Choose B, the number of resamples, to obtain desired accuracy

### General procedure
1. Compute statistic for original sample
2. Choose permutation resamples that are consistent with null hypothesis of test and study design. Construct permutation distribution from high number of resamples.
3. Find P-value by locating original statistic on permutation distribution

### Types of problems
* Two-sample problems
    * Null hypothesis: two populations are identical
    * Compare population means, proportions, standard deviations, etc.
* Matched pairs design
    * Null hypothesis: only random differences within pairs
    * Resample by permuting the two observations within each pair separately
* Relationships with other quantitative variables
    * Null hypothesis: variables not related
    * Randomly reassign values to two variables


## Resources
* [Bootstrap Methods and Permutation Testing Chapter](http://math.ntnu.edu.tw/~pwtsai/comp97/moore14.pdf)