# Algorithmic Fairness in ML
## Introduction
This repository implements several postprocessing algorithms designed to debias pretrained classifiers. It uses linear programming for everything, including its work with real-valued predictors, and so the bulk of the solving code is an implementation (and extension) of the method presented by Hardt, Price, and Srebro in their [2016 paper](https://arxiv.org/pdf/1610.02413.pdf) on fairness in supervised learning. 

## Methods
### Background
The main goal of any postprocessing method is to take an existing classifier and make it fair for all levels of a protected category, like race or religion. There are a number of ways to do this, but in Hardt, Price, and Srebro's paper, they take an oblivious approach, such that the adjusted classifier (or derived predictor, Y<sub>tilde</sub>) only relies on the joint distribution of the true label (Y), the predicted label (Y<sub>hat</sub>), and the protected attribute (A).

#### Discrete predictor for a binary outcome
When both Y and Y hat are binary, the optimization problem is a linear program based on the conditional probabilities (Y<sub>tilde</sub> = Y<sub>hat</sub>). In this case, two probabilities, (Y<sub>tilde</sub> = 1 | Y<sub>hat</sub> = 1) and (Y<sub>tilde</sub> = 1 | Y<sub>hat</sub> = 0) fully define the distribution of the derived predictor, and so the linear program can be written in a very parismonious 2 * N<sub>groups</sub> variables. The solution will find the topmost-left point of the intersection of the group-specific convex hulls in ROC space (illustration below).

<img src="https://github.com/scotthlee/fairness/blob/master/img/nolines.png" width="400" height="300"><img src="https://github.com/scotthlee/fairness/blob/master/img/lines.png" width="400" height="300">

#### Continuous predictor for a binary outcome
When Y is binary but Y<sub>hat</sub> is continuous, Y<sub>hat</sub> must be thresholded before we can examine it for fairness. Hardt, Price, and Srebrbo proposed adding randomness to the selection of threshold for each group to make the resulting predictions fair. Here, we take the arguably more straightforward approach of thresholding the scores first (choosing thresholds that maximize groupwise performance) and then using the linear program as in the discrete case above to solve for the derived predictor. Theoretically, this may be sub-optimal, but practically, it runs fast and works well (illustration below). 

<img src="https://github.com/scotthlee/fairness/blob/master/img/roc_nolines.png" width="400" height="300"><img src="https://github.com/scotthlee/fairness/blob/master/img/roc_lines.png" width="400" height="300">

#### Multiclass outcomes
Coming soon!

### Implementation
Our implementation relies on a single class, the `PredictionBalancer`, to perform the adjustment. Initializing the balancer with the true label, the predicted label, and the protected attribute will produce a report with the groupwise true- and false-positive rates. The rest of its functionality comes from a few key methods--see the class's [docstrings](balancers/__init__.py) for more info!

## Data
For demo purposes, the repository comes with a synthetic dataset, `farm_animals.csv`, which we created with `data_gen.py`. Here are the data elements:

1. `animal`: The kind of farm animal. Options are `cat`, `dog`, and `sheep`. This is the protected attribute A.
2. `action`: The kind of care the animal needs. Options are `feed`, `pet`, and `shear`. This is the true label Y.
3. `pred_action`: The kind of care the farmer thinks the animal needs. This is the predicted label Y<sub>hat</sub>.
4. `shear`: Whether `pred_action` is `shear`.
5. `shear_prob`: The predicted probability that the animal needs a shear. This was generated using different conditional probabilities than the variable `pred_action`, so it will not equal `shear` when thresholded. 

The distirbution of animals is not entirely realistic--a working sheep farmer, for example, would have a much higher ratio of sheep to herding dogs--but the lower class imbalance makes the demo a bit easier to follow.

## Demo
To see the postprocessing algorithms in action, please check out the [demo notebook](demo.ipynb). The notbeook shows off the main features of the `PredictionBalancer` and is the best place to start if you've never worked with these kinds of adjustments before. Please note: The modules in `requirements.txt` must be installed before running.

## Citation
If you use this code for a project, give us a shout out (DOI for citations below).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4890946.svg)](https://doi.org/10.5281/zenodo.4890946)

