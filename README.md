# fairness
## Introduction
This repository implements several postprocessing algorithms designed to debias pretrained classifiers. It uses linear programming for everything, including its work with real-valued predictors, and so the bulk of the solving code is an implementation (and extension) of the method presented by Hardt, Price, and Srebro in their [2016 paper](https://arxiv.org/pdf/1610.02413.pdf) on fairness in supervised learning. 

## The methods
### Background
The main goal of any postprocessing method is to take an existing classifier and make it fair for all levels of a protected category, like race or religion. There are a number of ways to do this, but in Hardt, Price, and Srebro's paper, they take an oblivious approach, such that the adjusted classifier (or derived predictor, Y tilde) only relies on the joint distribution of the true label (Y), the predicted label (Y hat), and the protected attribute (A). 

### Implementation
Our implementation relies on a single class, the `PredictionBalancer`, which uses a number of 

## The data
For demo purposes, the repository comes with a synthetic dataset, `farm_animals.csv`, which we created with `data_gen.py`. Here are the data elements:

1. `animal`: The kind of farm animal. Options are `cat`, `dog`, and `sheep`. This is the protected attribute A.
2. `action`: The kind of care the animal needs. Options are `feed`, `pet`, and `shear`. This is the true label Y.
3. `pred_action`: The kind of care the farmer thinks the animal needs. This is the predicted label Y hat.
4. `shear`: Whether `pred_action` is `shear`.
5. `shear_prob`: The predicted probability that the animal needs a shear. This was generated using different conditional probabilities than the variable `pred_action`, so it will not equal `shear` when thresholded. 

The distirbution of animals is not entirely realistic--a working sheep farmer, for example, would have a much higher ratio of sheep to herding dogs--but the lower class imbalance makes the demo a bit easier to follow.

## The demo
To see the postprocessing algorithms in action, please check out `demo.ipynb`. The notbeook shows off the main features of the `PredictionBalancer` and is the best place to start if you've never worked with these kinds of adjustments before. Please note: The modules in `requirements.txt` must be installed before running.
