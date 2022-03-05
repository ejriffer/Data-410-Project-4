# Data-410-Project-4

## Multiple Boosting Algorithm

One way to improve upon Gradient Boosting and XGBoost is to apply multiple boosting to a dataset. This recursive call allows models to continually improve, and become more accurate the more times you boost it. 

Below multiple boosting algorithms have been applied to the Concerte Compressive Strength dataset. The nested loop includes a regular lowess model, a gradient boosted model, an XGBoosted model, a twice boosted model, and a three times boosted model. 

```
# data
X = concrete[['cement','slag','ash','water', 'superplastic','coarseagg','fineagg', 'age']].values
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

    # boosted 2
    model_boosting = RandomForestRegressor(n_estimators=100,max_depth=3)
    yhat_b2 = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,2)
    mse_b2.append(mse(ytest,yhat_b2))

    # boosted 3
    model_boosting = RandomForestRegressor(n_estimators=100,max_depth=3)
    yhat_b3 = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting,3)
    mse_b3.append(mse(ytest,yhat_b3))


print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Boosted LWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
print('The Cross-validated Mean Squared Error for 2 Boosted is : '+str(np.mean(mse_b2)))
print('The Cross-validated Mean Squared Error for 3 Boosted is : '+str(np.mean(mse_b3)))
```

## LightGBM

LightGBM is a Microsoft gradient boosting technique. Instead of having a default algorithm (like XGBoost) that can be difficult to optimize LightGBM uses histograms to split the features into bins which can be easier to optimize.

2. (Research) Read about the LightGBM algorithm and include a write-up that explains the method in your own words. Apply the method to the same data set you worked on for part 1. 
