Homework 4

Meiying Chen

Oct. 22, 2019

DSC 440 Data Mining



Exercise  7.5

```
Algorithm: Mining the set of negatively correlated patterns
Input:
D, a data set of transactions
min_sup, minimal support
epsilon, negative pattern threshold
Output: all nagetive patterns
Method:
scan D and find frequent 1-items for each level; 
for	k = 2,···,K
	scan D to compute Corr for all candidates (1, k)-itemsets and (2, k)-itemsets;
	prune based on min_sup, epsilon;
	end
for h = 3,···,H
	for k = 2, · · · , K
	scan D to compute Corr measure for candidate itemsets in Q(h,k);
	prune based on min_sup, epsilon; 
	end
return Q(h,k)
```


Exercise 7.9

For any distance measure to be valid, there are three conditions to be considered:
1. For i = j, d[i, j] = 0
2. For all (i, j), d[i,j] = d[j,i]
3. For all (i,j,k), d[i,k] <= d[i,j] + d[j,k]

For the described pattern distance measure:
$$
For\ any\ P1 = P2, 
Pat\_Dist(P1, P2) = 1 - \frac {|T(P1) \cap T(P1)|} { |T(P1) \cup T(P1)|}
= 1 - 1 = 0 \tag1 
$$
$$
For\ any\ P1, P2,
Pat\_Dist(P1, P2) = 1 - \frac {|T(P1) \cap T(P2)|} { |T(P1) \cup T(P2)|}\\
 = 1 - \frac {|T(P2) \cap T(P1)|} { |T(P2) \cup T(P1)|} = Pat\_Dist(P2, P1) \tag2
$$
$$
For\ any\ P1, P2, P3, 
If \ Pat\_Dist(P1, P3) \leq  Pat\_Dist(P1, P2) + Pat_Dist(P2, P3)\\
Then\ \frac {|T(P1) \cap T(P3)|} { |T(P1) \cup T(P3)|} \geq \frac {|T(P1) \cap T(P2)|} { |T(P1) \cup T(P2)|} + \frac {|T(P2) \cap T(P3)|} { |T(P2) \cup T(P3)|} - 1\\
Assume\ P1 \leq P3 \leq P2, the\ equation\ will\ be:\\
\frac {Sup(P1)} { Sup(P3)} \geq \frac {Sup(P1)} { Sup(P2)} + \frac {Sup(P2)} { Sup(P3)} - 1\\
\frac {Sup(P1) - Sup(P2)} { Sup(P3)} \geq \frac {Sup(P1) - Sup(P2)} { Sup(P2)} \\ which\ is\ also\ true\ for\ other\ five\ assumptions \tag3
$$

Exercise 8.3
 The procedure a is known as post-pruning, and b is pre-pruning. Compare these two methods, we can see generally, post-pruning keeps more branches than pre-pruning. So, the post-pruning lowers the risk of under-fitting, in other words, it is more accurate. Besides, as the tress is fully grown, it is easier to select a threshold compared to the pre-pruning method.

Exercise 8.5
We can use BOAT(bootstrapped optimistic algorithm for tree construction), separating the large training set into small parts, and every part could fit into the memory. Each subset of the originally dataset will be used to generate a tree, and the resulting several trees will be aggregated. 
For example, if the dataset is separated into 10 parts, which part contains average n labels(n is pretty small under 100). And the memory needs to train a single train is 50\*100\*n = several MB, much smaller than 512MB.

Exercise 8.7
(a) We should also use the count of each generalized tuple to calculate our information gain or gain ratio.
(b) Here we choose gain ratio.
$$
Gain\_ratio = \frac {Gain}{SplitINFO} \\
SplitINFO=-\sum {\frac {n_i}{n} log \frac {n_i}{n}}
$$
Firstly we need to run go over all classes to find the root node:
 For department:
$$
 Gain = (1 - \frac {113}{165}^2 +  \frac {52}{165} ^2)= 0.0315\\
 SplitINFO = -( \frac {110}{165} log  \frac {110}{165} +  \frac {}{} \frac {31}{165} log  \frac {31}{165} +  \frac {14}{165} log  \frac {14}{165} +  \frac {10}{165} log  \frac {10}{165}) = 0.9636\\
 Gain\_ratio =  \frac {Gain}{SplitINFO} = 0.0327\\
$$
Similarly, we can calculate that Gain_ratio(Age) = 0.0253, Gain_ratio(Salary) = 0.2494
As Salary has the biggest gain ratio value, it is the root node.
We continue this procedure, util we construct the whole tree as follow:
![tree.jpg](C:\Users\dell\Desktop\cmy\tree.jpg.jpg)
 (c) P(X|Junior) = 113/165 * 23 / 113 * 49/113* 23/113  = 0.012303
 P(X|Senior) = 52/165* 8/53 * 1/53 * 40/52 = 0.0006904215
 Because P(X|Junior) > P(X|Senior), the classifier would see the person as Junior.

 Exercise 8.12
 Code: evaluation.py

---- roc curve ---
when threshold is  0.95
FPR, TPR =  0.0 0.2
when threshold is  0.85
FPR, TPR =  0.2 0.2
when threshold is  0.66
FPR, TPR =  0.2 0.6
when threshold is  0.6
FPR, TPR =  0.4 0.6
when threshold is  0.55
FPR, TPR =  0.4 0.8
when threshold is  0.51
FPR, TPR =  1.0 0.8
when threshold is  0.4
FPR, TPR =  1.0 1.0
---- confusion metrix ----
threshold is: [ 0.95  0.85  0.66  0.6   0.55  0.51  0.4 ]
TN, FP, FN, TP =  5 0 4 1
threshold is: [ 0.95  0.85  0.66  0.6   0.55  0.51  0.4 ]
TN, FP, FN, TP =  4 1 4 1
threshold is: [ 0.95  0.85  0.66  0.6   0.55  0.51  0.4 ]
TN, FP, FN, TP =  4 1 2 3
threshold is: [ 0.95  0.85  0.66  0.6   0.55  0.51  0.4 ]
TN, FP, FN, TP =  3 2 2 3
threshold is: [ 0.95  0.85  0.66  0.6   0.55  0.51  0.4 ]
TN, FP, FN, TP =  3 2 1 4
threshold is: [ 0.95  0.85  0.66  0.6   0.55  0.51  0.4 ]
TN, FP, FN, TP =  0 5 1 4
threshold is: [ 0.95  0.85  0.66  0.6   0.55  0.51  0.4 ]
TN, FP, FN, TP =  0 5 0 5

![ROC](C:\Users\dell\Desktop\cmy\ROC.png)

Exercise 8.14
Code: model_select.py
We will be performing a t-test.
Null  hypothesis: M1 and M2 are the same.
t = (err(M1) - err(M2)) / sqrt(var(M1 - M2) / k) = 2.43756714123
With significant level = 1%, consult to the table and we get z = 3.250 > t
So, hypothesis is rejected. M1 and M2 are statistical significantly different.
Because mean(M1) = 27.72 > mean(M2) = 21.27, so M2 is better.