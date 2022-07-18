# Multiple Inference

Testing multiple variables increases the probability of making type I errors. Specifically, the probability of making at least one type I error when testing $n$ true null hypotheses is $1-(1-\alpha)^{n}$. On average, $\alpha n$ will result in falsely significant results.


## Family-Wise Error Rate
Family-wise error rate (FWER) is the probability of falsely rejecting at least one ture null hypothesis: $P[V \ge 1]$.

A procedure is said to control FWER at level $\alpha$ if $P[V \ge 1] \le \alpha$.


### Bonferroni Correction
The Bonferroni method controls FWER: the goal is to ensure that $P(\text{reject any null hypothesis}) \le \alpha$. This is achieved by performing each test at significance level $\alpha / n$ instead of $\alpha$. 

$$
P[\text{reject any null hypothesis}] 
$$

$$
= P[(\text{reject } H_{0}^{(1)}) \cup \cdots \cup (\text{reject } H _{0} ^{(n)})]
$$

$$
\le P[\text{reject } H_{0} ^{(1)}] + \dots + P[\text{reject } H_{0} ^{(n)}]
$$

$$
= \frac{\alpha} {n} + \dots + \frac{\alpha} {n} = \alpha
$$

In terms of $p$-values, the Bonferroni correction method rejects the null hypotheses whose corresponding $p$-values are at most $\alpha / n$.

A disadvantage of controlling the FWER is that it may greatly reduce the power of the test to detect real effects, especially when $n$, the number of tested hypotheses, is large.


## False Discovery Rate
False-discovery proportion (FDP) is defined by

$$
FDP = \begin{cases}
\frac{V}{R}, & R \ge 1 \\[2ex]
0, & R = 0
\end{cases}
$$

False-discovery rate (FDR) is the expected value of the FDP, $E(\text{FDP})$.

In comparison to FWER, FDR allows toleratio n of some type I errors, as long as most of the discoveries made are true.


### The Benjamini-Hochberg Procedure
The Benjamini-Hochberg (BH) procedure compares sorted $p$-values to a diagnoal cutoff line. It finds the largest $p$-value that falls below this line, and rejects the null hypotheses for all $p$-values up to this one:
1. Sort p-values so that $P_{(1)} \le \dots \le P_{(n)}$
2. Find the largest $r$ such that $P_{(r)} \le \frac{qr}{n}$
3. Reject null hypotheses $H_{0} ^{(1)}, \dots , H_{0} ^{(r)}$

#### Theorem (Benjamini and Hochberg (1995))
Consider tests of $n$ null hypotheses, $n_0$ of which are true. If the test statistics of these tests are independent, then the FDR of the described procedure satisfies

$$
\text{FDR} \le \frac {n_{0} q} {n} \le q
$$


## Resources
* [Introduction to Multiple Testing](https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture11.pdf)
* [Multiple Inference Mistakes](https://web.ma.utexas.edu/users/mks/statmistakes/multipleinference.html)