### Take-home Quiz

### Meiying Chen

### Oct.18, 2019

### CSC 442 Artificial Intelligence



#### The labyrinth from the last three slides

1.  Represent as Propositional Logic:

$$
\neg (G \and (S \Rightarrow M))
$$

$$
\neg(\neg G \and \neg S)
$$

$$
\neg(G \and \neg M)
$$



2. Conver to CMF:

   Rule 1:
   $$
   \neg (G \and (S \Rightarrow M)) \equiv \neg (G \and (\neg S \or M))\\\equiv \neg(( G \and \neg S) \or (G \and M))\\\equiv \neg( G \and \neg S) \and \neg (G \and M)\\\equiv (\neg G \or  S) \and (\neg G \or \neg M)\\result:\ [\neg G \or  S],[\neg G \or \neg M]
   $$
   Rule 2:

$$
\neg(\neg G \and \neg S) \equiv G \and S\\
result:[G, S]
$$

​		 Rule 3:
$$
\neg(G \and \neg M) \equiv \neg G\or M\\
result:[\neg G\or M]
$$
​		Put together:
$$
[\neg G \or  S]\tag1
$$

$$
[\neg G \or \neg M]\tag2
$$

$$
[G, S]\tag3
$$

$$
[\neg G\or M] \tag4
$$

3. Solve the puzzle in the form of an resolutional proof

   Stone road is a good choice:
   $$
   [\neg S]\tag5
   $$

   $$
   [S] \ ((1) \and (3))\tag6
   $$

   $$
   [] \ ((5)\and(6))\tag7
   \\\Delta \and \neg S \ is\ unsatisfitable\\
   So, \Delta \models S
   $$

#### Problem 7.6 form textbook

(a) For 'and': True, because any  model in gamma, it is in model alpha and in model beta at the same time.

Resolution:
$$
(\neg \alpha \models \gamma) \or (\neg \beta \models \gamma) \equiv \neg( \alpha \and \neg \gamma) \or \neg( \beta \and \neg \gamma) \\
\equiv ( \neg \alpha \and \gamma) \or (\neg \beta \and  \gamma)\\
\equiv  \neg \alpha \and \gamma \and \neg \beta \and  \gamma\\
result:[\neg \alpha,\neg \beta, \gamma]\tag1
$$

$$
\neg(\neg((\alpha \and \beta)\and \neg \gamma)) \equiv (\alpha \and \beta)\and \neg \gamma\\
\equiv \alpha \and \beta \and \neg \gamma\\
result: [ \alpha, \beta , \neg \gamma]\tag2\\
\\
(1) \and (2) \ is \ [],\\
\\\Delta \and \neg ((\alpha \and \beta) \models \gamma) \ is\ unsatisfitable\\
So, \Delta \models ((\alpha \and \beta) \models \gamma)
$$

For 'or': True, because for any model in gamma, it is either in model alpha or in model beta.

Resolution:
$$
(\neg \alpha \models \gamma) \and (\neg \beta \models \gamma) \equiv \neg( \alpha \and \neg \gamma) \and \neg( \beta \and \neg \gamma) \\\equiv ( \neg \alpha \and \gamma) \and (\neg \beta \and  \gamma)\\result:[\neg \alpha],[\neg \beta], [\gamma]\tag1\\
$$

$$
\neg(\neg((\alpha \and \beta)\and \neg \gamma)) \equiv (\alpha \and \beta)\and \neg \gamma\\\equiv \alpha \and \beta \and \neg \gamma\\result: [ \alpha, \beta , \neg \gamma]\tag2\\\\(1) \and (2) \ is \ [],\\\\\Delta \and \neg ((\alpha \and \beta) \models \gamma) \ is\ unsatisfitable\\So, \Delta \models ((\alpha \and \beta) \models \gamma)
$$



(b) True, because every model in (beta and gamma) is in alpha, beta and gamma.

Resolution:
$$
\alpha \models (\beta \and \gamma) \equiv \neg(\alpha \and \neg(\beta \and \gamma))\\
\equiv \neg \alpha \and (\beta \or \gamma))\\
\equiv (\neg \alpha \and  \beta ) \or (\neg \alpha  \and \gamma))\\
result: [neg \alpha,  \beta], [\neg \alpha  , \gamma]\tag1\\
$$

$$
\neg (\alpha \models \beta \and \alpha \models \ gamma) \equiv \neg (\neg (\alpha \and \beta) \and \neg (\alpha \and \neg \gamma)\\
\equiv (\alpha \and \neg \beta) \or (\alpha \and \neg \gamma)\\
result:[(\alpha,  \neg \beta)],[(\alpha , \neg \gamma)]\tag2\\

\\(1) \and (2) \ is \ [],\\\\\Delta \and \neg (\alpha  \models \gamma \and \beta \models \gamma) \ is\ unsatisfitable\\So, \Delta \models (\alpha  \models \gamma \and \beta \models \gamma)
$$

(3) False. For example, when:
$$
\beta = \neg \gamma
$$
















