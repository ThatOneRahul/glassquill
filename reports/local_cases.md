# Local explanations (XGBoost + SHAP)

## true_ai (P(AI) = 1.00, source: reddit_eli5)

- Unevenness of surprise = 1.97 pushed the score towards AI
- Perplexity (log) = 2.28 pushed the score towards AI
- Share of top-10 guessable words = 0.84 pushed the score towards AI

> When you search for something on the internet using Google, Google is able to search through billions of web pages very quickly because it has built a special database just for searching the internet. This database is stored on many computers all around the world, which helps Google search the internet even faster. When you search for something on your computer's hard drive, your computer has to look through all of the files and folders on your hard drive to find what you're looking for. This ca...

## true_human (P(AI) = 0.00, source: reddit_eli5)

- Average word length = 4.41 pushed the score towards human
- Commas per word = 0.04 pushed the score towards human
- Perplexity (log) = 3.23 pushed the score towards human

> There's actually not much point to it. The cruising altitude for a commercial jet is well out of the parachuting range. If the doors to an airplane were opened while you were at cruising altitude, everyone would either be sucked out of the plane, or would lose consciousness because there's such a small amount of air up there. So if a plane was in the middle of crashing, it would only make sense to use a parachute when the plane is into the safe parachute altitude range. And trying to organize a ...

## false_positive (P(AI) = 0.98, source: wiki_csai)

- Average word length = 7.06 pushed the score towards AI
- Sentence length spread = 5.31 pushed the score towards AI
- Function word rate = 0.20 pushed the score towards human

> Stochastic optimization (SO) methods are optimization methods that generate and use random variables. For stochastic problems, the random variables appear in the formulation of the optimization problem itself, which involves random objective functions or random constraints. Stochastic optimization methods also include methods with random iterates. Some stochastic optimization methods use random iterates to solve stochastic problems, combining both meanings of stochastic optimization. Stochastic ...

## false_negative (P(AI) = 0.38, source: reddit_eli5)

- Average word length = 4.28 pushed the score towards human
- Unevenness of surprise = 2.27 pushed the score towards AI
- Sentence length spread = 3.54 pushed the score towards AI

> Cable stations often play the same movie twice in a row on weekends because it can be a good way to fill airtime and attract viewers. Some people might enjoy watching the same movie twice, especially if it is a favorite movie or if they missed it the first time it was shown. Playing two movies with overlapping audiences might also be a good idea, but it can be difficult to predict which movies will have the most appeal to viewers. It is ultimately up to the cable stations to decide which movies ...
