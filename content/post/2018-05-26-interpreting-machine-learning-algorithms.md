---
title: "Interpreting Machine Learning Algorithms"
date: 2018-05-26T21:26:57-05:00
categories:
- Link
- Machine Learning
- Black Box
- Algorithm Interpretation
comments: true
---

I've had an open tab with an [overview piece on interpreting machine learning algorithms](https://www.oreilly.com/ideas/ideas-on-interpreting-machine-learning) for several weeks now. After finally reading it, I grabbed a few useful quotes and provided them below:

> Most machine learning algorithms create nonlinear, non-monotonic response functions. This class of functions is the most difficult to interpret, as they can change in a positive and negative direction and at a varying rate for any change in an independent variable. Typically, the only standard interpretability measure these functions provide are relative variable importance measures.

> Variable importance measures rarely give insight into even the average direction that a variable affects a response function. They simply state the magnitude of a variable's relationship with the response as compared to other variables used in the model.

Also, this quote stood out to me more than the others. As data scientists and ML practitioners, we are trained to **trust** a model by understanding its implementation details and various holdout validation scores. But for most end-users, the bar is much higher.

> For some users, technical descriptions of algorithms in textbooks and journals provide enough insight to fully understand machine learning models. For these users, cross-validation, error measures, and assessment plots probably also provide enough information to trust a model. Unfortunately, for many applied practitioners, the usual definitions and assessments don't often inspire full trust and understanding in machine learning models and their results. The techniques presented here go beyond the standard practices to engender greater understanding and trust. 
