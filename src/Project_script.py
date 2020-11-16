# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 23:24:50 2020

@author: Cornellius
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import model_selection
from scipy import stats
from toolbox_02450 import rlr_validate
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from toolbox_02450 import train_neural_net, draw_neural_net

#=====ANALYSIS_SETUP
doBaseline=True
doLinearModel=True
doRLRModel=True
doANN=True

#=====DataLoadingSetup
filename = '../data/Real_estate_valuation_data_set.csv'
dataset = pd.read_csv(filename)
raw_data = dataset.to_numpy()
attributeNames = list(dataset)[2:7]
cols = range(2, 7) 
X = raw_data[:, cols].astype(np.float)
y = raw_data[:, 7].astype(np.float)
cols_ANN = range(2, 7) 
X_ANN_base = raw_data[:, cols].astype(np.float)
Y_ANN_base = raw_data[:,[7]].astype(np.float)
N, M = X.shape
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

#====CrossValidationSetup    
K = 10
CV = model_selection.KFold(K, shuffle=True)
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

#====BaseLineSetup
if doBaseline==True:
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    Error_test_baseline = np.empty((K,1))
#====LinearModelSetup
if doLinearModel==True:
    Error_train_lm = np.empty((K,1))
    Error_test_lm = np.empty((K,1))
    w_noreg = np.empty((M,K))

#====RegularizedLinearModelSetup
if doRLRModel==True:
    lambdas = np.power(10.,range(-5,9))
    Error_train_rlr = np.empty((K,1))
    Error_test_rlr = np.empty((K,1))
    w_rlr = np.empty((M,K))
    optimalLambdas = np.empty((K,1))
#====ANNSetup

    
best_ann_h_error = np.empty((K,2))
j=0
for train_index, test_index in CV.split(X,y):    

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    X_ANN_train = X_ANN_base[train_index]
    X_ANN_test = X_ANN_base[test_index]
    y_ann_train = Y_ANN_base[train_index]
    y_ann_test = Y_ANN_base[test_index]
    
    X_ANN_train = stats.zscore(X_ANN_train[:, 0:])
    X_ANN_test = stats.zscore(X_ANN_test[:, 0:])
    
    mu[j, :] = np.mean(X_train[:, 1:], 0)
    sigma[j, :] = np.std(X_train[:, 1:], 0)
    X_train[:, 1:] = stats.zscore(X_train[:, 1:])
    X_test[:, 1:] = stats.zscore(X_test[:, 1:])
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    if doBaseline==True:
        Error_train_nofeatures[j] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        Error_test_nofeatures[j] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
        Error_test_baseline[j] = np.square(y_test-y_train.mean()).sum(axis=0)/y_test.shape[0]
    if doLinearModel==True:
        w_noreg[:,j] = np.linalg.solve(XtX,Xty).squeeze()
        Error_train_lm[j] = np.square(y_train-X_train @ w_noreg[:,j]).sum(axis=0)/y_train.shape[0]
        Error_test_lm[j] = np.square(y_test-X_test @ w_noreg[:,j]).sum(axis=0)/y_test.shape[0]
    
    if doRLRModel==True:
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, K)
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0,0] = 0
        w_rlr[:,j] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        Error_train_rlr[j] = np.square(y_train-X_train @ w_rlr[:,j]).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[j] = np.square(y_test-X_test @ w_rlr[:,j]).sum(axis=0)/y_test.shape[0]
        optimalLambdas[j] = opt_lambda
        
    if doANN==True:
        
        n_replicates = 1        # number of networks trained in each k-fold
        max_iter = 10000
        CV_ANN = model_selection.KFold(K, shuffle=True)
        X_ANN = X_ANN_train
        y_ANN = y_ann_train
        # Setup figure for display of learning curves and error rates in fold
        summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
        # Make a list for storing assigned color of learning curve for up to K=10
        color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                      'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
        mean_error_array=np.empty((4,1))
        # Define the model
        for counter in range(1,5):
            n_hidden_units = counter
            # number of hidden units
            model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M-1, n_hidden_units), #M features to n_hidden_units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                        # no final tranfer function, i.e. "linear output"
                        )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
            print('Training model of type:\n\n{}\n'.format(str(model())))
            errors = [] # make a list for storing generalizaition error in each loop
            for (k, (train_index_2, test_index_2)) in enumerate(CV_ANN.split(X_ANN,y_ANN)):
                print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
        
                # Extract training and test set for current CV fold, convert to tensors
                X_train_ann = torch.Tensor(X_ANN[train_index_2,:])
                y_train_ann = torch.Tensor(y_ANN[train_index_2])
                X_test_ann = torch.Tensor(X_ANN[test_index_2,:])
                y_test_ann = torch.Tensor(y_ANN[test_index_2])
                
                net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train_ann,
                                                           y=y_train_ann,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
                print('\n\tBest loss: {}\n'.format(final_loss))
                # Determine estimated class labels for test set
                y_test_est = net(X_test_ann)
                # Determine errors and errors
                se = (y_test_est.float()-y_test_ann.float())**2 # squared error
                mse = (sum(se).type(torch.float)/len(y_test_ann)).data.numpy() #mean
                errors.append(mse) # store error rate for current CV fold 
                # Display the learning curve for the best net in the current fold
                h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
                h.set_label('CV fold {0}'.format(k+1))
                summaries_axes[0].set_xlabel('Iterations')
                summaries_axes[0].set_xlim((0, max_iter))
                summaries_axes[0].set_ylabel('Loss')
                summaries_axes[0].set_title('Learning curves')
            mean_error_array[counter-1] = sum(errors)/len(errors)
            # Display the MSE across folds
            summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
            summaries_axes[1].set_xlabel('Fold')
            summaries_axes[1].set_xticks(np.arange(1, K+1))
            summaries_axes[1].set_ylabel('MSE')
            summaries_axes[1].set_title('Test mean-squared-error')
            print('Diagram of best neural net in last fold:')
            weights = [net[i].weight.data.numpy().T for i in [0,2]]
            biases = [net[i].bias.data.numpy() for i in [0,2]]
            tf =  [str(net[i]) for i in [1,2]]
            draw_neural_net(weights, biases, tf, attribute_names=attributeNames[1:])  
            # Print the average classification error rate
            print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
            
            # When dealing with regression outputs, a simple way of looking at the quality
            # of predictions visually is by plotting the estimated value as a function of 
            # the true/known value - these values should all be along a straight line "y=x", 
            # and if the points are above the line, the model overestimates, whereas if the
            # points are below the y=x line, then the model underestimates the value
            plt.figure(figsize=(10,10))
            y_est = y_test_est.data.numpy(); 
            y_true = y_test_ann.data.numpy()
            axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
            plt.plot(axis_range,axis_range,'k--')
            plt.plot(y_true, y_est,'ob',alpha=.25)
            plt.legend(['Perfect estimation','Model estimations'])
            plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
            plt.ylim(axis_range); plt.xlim(axis_range)
            plt.xlabel('True value')
            plt.ylabel('Estimated value')
            plt.grid()
    
            plt.show()

        best_ann_h_error[j,0] = np.min(mean_error_array)
        
        best_ann_h_error[j,1] = np.argmin(mean_error_array)+1
#=====================================PLOTTING
    if doRLRModel==True:
        figure(j, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    print('Cross validation fold {0}/{1}:'.format(j+1,K))
    print('Train indices: {0}'.format(train_index))
    print('Test indices: {0}\n'.format(test_index))
    
    
    j+=1
    

# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train_lm.mean()))
print('- Test error:     {0}'.format(Error_test_lm.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_lm.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_lm.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold (rlr):')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
    
print('Weights in last fold (lm):')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_noreg[m,-1],2)))
    
#============================== ANN
 
