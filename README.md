# Data-410-Project-4

## Multiple Boosting Algorithm

One way to improve upon Gradient Boosting and XGBoost is to apply multiple boosting to a dataset. This recursive call allows models to continually improve, and become more accurate the more times you boost it. 

Below multiple boosting algorithms have been applied to the Concerte Compressive Strength dataset. The nested loop includes a regular lowess model, a gradient boosted model, an XGBoosted model, a twice boosted model, and a three times boosted model. 

1. Create your own multiple boosting algortihm and apply it to combinations of different regressors (for example you can boost regressor 1 with regressor 2 a couple of times) on the "Concrete Compressive Strength" dataset.  Show what was the combination that achieved the best cross-validated results.

## LightGBM

LightGBM is a Microsoft gradient boosting technique. Instead of having a default algorithm (like XGBoost) that can be difficult to optimize LightGBM uses histograms to split the features into bins which can be easier to optimize.

2. (Research) Read about the LightGBM algorithm and include a write-up that explains the method in your own words. Apply the method to the same data set you worked on for part 1. 
