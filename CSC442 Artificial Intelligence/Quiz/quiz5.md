###Quiz 5

###CSC 442 Artificial Intelligence

###Meiying Chen

###Nov. 11, 2019



###13.1

$$
P(a|b \and a)= \frac{P(a\and (b\and a))}{P(b \and a)} = \frac{P(a\and b)}{P(b \and a)}=1
$$



###13.3

(a) True
$$
P(a|b,c) = P(b|a,c)\\
\frac{P(a,b,c)}{P(b,c)} = \frac{P(a,b,c)}{P(a,c)}\\
P(a,c) = P(b,c)\\
\frac{P(a,c)}{P(c)}=\frac{P(b,c)}{P(c)}\\
P(a|c) = P(b|c)
$$
(b) False

Because P(a|b,c) = P(a) only implies that a is independent of (b,c), and it dose not imply anything about the relationship between b and c. To give an example, if a and b is independent like flip a coin, and c is dependent to b like c is the flip side of b's result, then the statement is false.

(c) False

P(a|b)=P(a) only implies a is independent of b, but not an evident of a is conditionally independent of b given c. The statement is false if a and b is result of flipping a coin for two times independently, and c equals a + b.



### 13.10

(a)   
$$
Expected\ payback=(\frac{1}{4})^3*20+(\frac{1}{4})^3*15+(\frac{1}{4})^3*5+(\frac{1}{4})^3*3+(\frac{1}{4})^2*\frac{3}{4}*2  +\frac{1}{4}*\frac{3}{4}*1=\frac{61}{64}
$$


(b)
$$
Chance\ of\ win =(\frac{1}{4})^3+(\frac{1}{4})^3+(\frac{1}{4})^3+(\frac{1}{4})^3+(\frac{1}{4})^2*\frac{3}{4}  +\frac{1}{4}*\frac{3}{4}*1=\frac{19}{64}
$$


