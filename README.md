# Data-410-Project-4

## Multiple Boosting Algorithm

One way to improve upon Gradient Boosting and XGBoost is to apply multiple boosting to a dataset. This recursive call continually boosts models that have already been boosted in hopes of increasingly improving the accuracy. 

Below multiple boosting algorithms have been applied to the Concerte Compressive Strength dataset. The nested loop includes a regular lowess model, a gradient boosted model, an XGBoosted model, a twice boosted model, and a three times boosted model for comparison. 

```
# data
X = concrete[['cement', 'slag', 'superplastic']].values
y = concrete['strength'].values

mse_lwr = []
mse_blwr = []
mse_xgb = []
mse_b2 = []
mse_b3 = []

for i in range(1):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = X[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = X[idxtest]
    scale = StandardScaler()
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)
    dat_train = np.concatenate([xtrain,ytrain.reshape(-1,1)],axis=1)
    dat_test = np.concatenate([xtest,ytest.reshape(-1,1)],axis=1)

    # normal lowess call
    yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
    mse_lwr.append(mse(ytest,yhat_lwr))

    # normal boosted
    yhat_blwr = reg_boosted_lwr(xtrain,ytrain, xtest,Tricubic,tau=1.2,intercept=True)
    mse_blwr.append(mse(ytest,yhat_blwr))

    # normal XGB
    model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=1)
    model_xgb.fit(xtrain,ytrain)
    yhat_xgb = model_xgb.predict(xtest)
    mse_xgb.append(mse(ytest,yhat_xgb))

    # 2 boosted
    model_boosting = RandomForestRegressor(n_estimators=100,max_depth=3)
    yhat_b2 = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
    mse_b2.append(mse(ytest,yhat_b2))

    # 3 boosted
    model_boosting = RandomForestRegressor(n_estimators=100,max_depth=3)
    yhat_b3 = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,3)
    mse_b3.append(mse(ytest,yhat_b3))

print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Boosted LWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
print('The Cross-validated Mean Squared Error for 2 Boosted is : '+str(np.mean(mse_b2)))
print('The Cross-validated Mean Squared Error for 3 Boosted is : '+str(np.mean(mse_b3)))
```

Based on the code above we get the following outputs:

~ The Cross-validated Mean Squared Error for LWR is : 150.76979944789622

~ The Cross-validated Mean Squared Error for Boosted LWR is : 149.4101861333482

~ The Cross-validated Mean Squared Error for XGB is : 153.6022898582568

~ The Cross-validated Mean Squared Error for 2 Boosted is : 147.08542734363533

~ The Cross-validated Mean Squared Error for 3 Boosted is : 147.5077686091085

We can conclude that the repeated boosting models are the best, with the 2 boosted model performing just slighly better than the 3 boosted model.

## LightGBM

LightGBM is a Microsoft gradient boosting technique. It uses a decision tree to boost the outcome like normal gradient boosting and XGBoosting. However, instead of having a default algorithm (like XGBoost) that can be difficult to optimize LightGBM uses histograms to split the features into bins. 

LightGBM has a few advantages over other gradient boosting techniques including a faster run time and a higher accuracy. It also uses less memory and can handle large-sale data with efficiency. 

Below LightGBM has been applied to the same Concerte Compressive Strength dataset as seen above. 

```
mse_gbm = []

kf = KFold(n_splits=10,shuffle=True,random_state=100)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)  
  
  gbm = lgb.LGBMRegressor()
  gbm.fit(xtrain, ytrain, eval_set=[(xtest, ytest)], eval_metric='l1', early_stopping_rounds=1000)
  yhat_gbm = gbm.predict(xtest, num_iteration=gbm.best_iteration_)
  mse_gbm.append(mse(ytest,yhat_gbm))
  
print('The Cross-validated Mean Squared Error for GBM is : '+str(np.mean(mse_gbm)))
```
Based on the code above we get the following output:

~ The Cross-validated Mean Squared Error for GBM is : 144.1742765861046

LightGBM outperformed every model before, as well as running much quicker. Therefore we can conlucde the LightGBM is the most accurate model for this data. 
