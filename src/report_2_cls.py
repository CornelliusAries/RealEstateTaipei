#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Report 2
#Author: MH

import xlrd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import copy

# Load xls sheet with data
dataSet = xlrd.open_workbook('../Data/Real_estate_valuation_data_set.xlsx').sheet_by_index(0)
M=len(dataSet.row_values(0, 2, 8))
N = 414
X = np.empty((N, 5))
for i, col_id in enumerate(range(2, 7)):
    X[:, i] = np.asarray(dataSet.col_values(col_id, 1, N+1))
    
y = dataSet.col_values(7, 1, N+1)

# Deleting the outlier
X=np.delete(X,np.where(y==np.max(y))[0][0],0)
y=np.delete(y,np.where(y==np.max(y))[0][0],0)
N -= 1
# Setting number of shops as 'y' variable in a classification problem
buf = X[:, 2].astype(int)
X[:, 2] = copy.copy(y)
y = buf
C = max(y)+1

# Normalize the data
mu = np.mean(X, 0)
sigma = np.std(X, 0)
X = (X - mu) / sigma

#%% Baseline model
count_mat = np.empty((C,2), dtype=int)
for i in range(C):
    count_mat[i,:] = [i, sum(y==i)]
print('Baseline error: {}'.format(1-max(count_mat[:,1])/sum(count_mat[:,1])))


#%% Logistic regression model
K = 10
lambda_interval = np.logspace(-3, 0, 100)
lambda_l = len(lambda_interval)
Error_train = np.empty((K,lambda_l))
Error_test = np.empty((K,lambda_l))
Error_train_bs = np.empty((K,lambda_l))
Error_test_bs = np.empty((K,lambda_l))

k=0
for train_index, test_index in model_selection.KFold(n_splits=K,shuffle=True).split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    X_train_bs, X_test_bs = np.delete(X_train,2,1), np.delete(X_test,2,1)
    
    for l in range(lambda_l):
        # Logistic regression model
        logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, C=1/lambda_interval[l], max_iter=1000)
        logreg.fit(X_train,y_train)
        logreg_bs = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, C=1/lambda_interval[l], max_iter=1000)
        logreg_bs.fit(X_train_bs,y_train)
        y_est_test = logreg.predict(X_test)
        y_est_train = logreg.predict(X_train)
        y_est_test_bs = logreg_bs.predict(X_test_bs)
        y_est_train_bs = logreg_bs.predict(X_train_bs)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        Error_test[k,l] = np.sum(y_est_test != y_test) / float(len(y_est_test))
        Error_train[k,l] = np.sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test_bs[k,l] = np.sum(y_est_test_bs != y_test) / float(len(y_est_test_bs))
        Error_train_bs[k,l] = np.sum(y_est_train_bs != y_train) / float(len(y_est_train_bs))
        # Number of miss-classifications
        #print('Number of miss-classifications:\n\t {0} out of {1}'.format(np.sum(y_est_test != y_test),len(y_test)))
        #print('Generalization error for lambda = {0}: {1}'.format(lambda_interval[l],Error_test[k,:].mean()))

    k+=1

plt.figure()
plt.plot(lambda_interval, np.mean(Error_train, 0),'--b')
plt.plot(lambda_interval, np.mean(Error_test, 0),'--r')
plt.plot(lambda_interval, np.mean(Error_train_bs, 0),'-b')
plt.plot(lambda_interval, np.mean(Error_test_bs, 0),'-r')
plt.xscale('log')
plt.xlabel('lambda (-)')
plt.ylabel('error (-)')
plt.legend({'training set full model','test set full model','training set reduced model','test set reduced model'})
plt.show()
print('The smallest test error for full model is {0} for the lambda {1}'.format(min(np.mean(Error_test, 0)), lambda_interval[np.argmin(np.mean(Error_test, 0))]))
print('The smallest test error for reduced model is {0} for the lambda {1}'.format(min(np.mean(Error_test_bs, 0)), lambda_interval[np.argmin(np.mean(Error_test_bs, 0))]))


#%% LR continue
K = 10
lambda_val = 0.04977
Error_train = np.empty((K,M))
Error_test = np.empty((K,M))

k=0
for train_index, test_index in model_selection.KFold(n_splits=K,shuffle=True).split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
    
    for l in range(M):
        if l == M-1:
            X_train_l, X_test_l = X_train, X_test
        else:
            X_train_l, X_test_l = np.delete(X_train,l,1), np.delete(X_test,l,1)
            
        # Logistic regression model
        logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, C=1/lambda_val, max_iter=1000)
        logreg.fit(X_train_l,y_train)
        y_est_test = logreg.predict(X_test_l)
        y_est_train = logreg.predict(X_train_l)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        Error_test[k,l] = np.sum(y_est_test != y_test) / float(len(y_est_test))
        Error_train[k,l] = np.sum(y_est_train != y_train) / float(len(y_est_train))
        # Number of miss-classifications
        #print('Number of miss-classifications:\n\t {0} out of {1}'.format(np.sum(y_est_test != y_test),len(y_test)))
        #print('Generalization error for lambda = {0}: {1}'.format(lambda_interval[l],Error_test[k,:].mean()))

    k+=1
    
plt.figure()
plt.plot(np.mean(Error_train, 0))
plt.plot(np.mean(Error_test, 0))
plt.xlabel('Model no.')
plt.ylabel('error (-)')
plt.legend({'training set','test set'})
plt.show()
print('The smallest test error {0} is for the lambda {1}'.format(min(np.mean(Error_test, 0)), np.argmin(np.mean(Error_test, 0))))

#%% LR continue2
K = 10
lambda_val = 0.04977
Error_train = np.empty((K,M-1))
Error_test = np.empty((K,M-1))

k=0
for train_index, test_index in model_selection.KFold(n_splits=K,shuffle=True).split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = np.delete(X,2,1)[train_index,:], y[train_index]
    X_test, y_test = np.delete(X,2,1)[test_index,:], y[test_index]
    
    for l in range(M-1):
        if l == M-2:
            X_train_l, X_test_l = X_train, X_test
        else:
            X_train_l, X_test_l = np.delete(X_train,l,1), np.delete(X_test,l,1)
            
        # Logistic regression model
        logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, C=1/lambda_val, max_iter=1000)
        logreg.fit(X_train_l,y_train)
        y_est_test = logreg.predict(X_test_l)
        y_est_train = logreg.predict(X_train_l)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        Error_test[k,l] = np.sum(y_est_test != y_test) / float(len(y_est_test))
        Error_train[k,l] = np.sum(y_est_train != y_train) / float(len(y_est_train))
        # Number of miss-classifications
        #print('Number of miss-classifications:\n\t {0} out of {1}'.format(np.sum(y_est_test != y_test),len(y_test)))
        #print('Generalization error for lambda = {0}: {1}'.format(lambda_interval[l],Error_test[k,:].mean()))

    k+=1
    
plt.figure()
plt.plot(np.mean(Error_train, 0))
plt.plot(np.mean(Error_test, 0))
plt.xlabel('Model no.')
plt.ylabel('error (-)')
plt.legend({'training set','test set'})
plt.show()
print('The smallest test error {0} is for the lambda {1}'.format(min(np.mean(Error_test, 0)), np.argmin(np.mean(Error_test, 0))))


#%% KNN model
K = 20
Error_train = np.zeros(K)
Error_test = np.zeros(K)
X_knn = X[:,[3,4]]

count=1
for train_index, test_index in model_selection.LeaveOneOut().split(X):
    print('Computing for split = {0}/{1}..'.format(count,N))
    # extract training and test set for current fold
    X_train, y_train = X_knn[train_index,:], y[train_index]
    X_test, y_test = X_knn[test_index,:], y[test_index]
    
    for k in range(K):
        # KNN model
        knclassifier = KNeighborsClassifier(n_neighbors=k+1)
        knclassifier.fit(X_train, y_train)
        y_est_test = knclassifier.predict(X_test)
        y_est_train = knclassifier.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        Error_test[k] += (y_est_test != y_test)
        Error_train[k] += np.sum(y_est_train != y_train)/(N-1)
        # Number of miss-classifications
        #print('Number of miss-classifications:\n\t {0} out of {1}'.format(np.sum(y_est_test != y_test),len(y_test)))
    count+=1

Error_test /= N
Error_train /= N

plt.figure()
plt.plot(list(range(1,K+1)), Error_train)
plt.plot(list(range(1,K+1)), Error_test)
plt.xlabel('K parameter')
plt.ylabel('error (-)')
plt.legend({'training set','test set'})
plt.show()
print('The smallest test error {0} is for the K = {1}'.format(min(Error_test), np.argmin(Error_test)+1))


#%% Naive-Bayes model
alpha = 1
Error_train = np.empty(N)
Error_test = np.empty(N)

X_nb = X[:,[3,4]]
x_dif = np.max(X_nb, 0)-np.min(X_nb, 0)
X_nb = (X_nb-np.ones((N,1))*np.min(X_nb,0))/(np.ones((N,1))*x_dif)

k=0
for train_index, test_index in model_selection.LeaveOneOut().split(X_nb):
    print('Computing for k = {0}/{1}..'.format(k+1,N))
    # extract training and test set for current fold
    X_train, y_train = X_nb[train_index,:], y[train_index]
    X_test, y_test = X_nb[test_index,:], y[test_index]
    # Naive-Bayes model
    nb_classifier = MultinomialNB(alpha=alpha, fit_prior=True)
    nb_classifier.fit(X_train, y_train)
    y_est_prob_test = nb_classifier.predict_proba(X_test)
    y_est_prob_train = nb_classifier.predict_proba(X_train)
    y_est_test = np.argmax(y_est_prob_test,1)
    y_est_train = np.argmax(y_est_prob_train,1)
    # Evaluate misclassification rate over train/test data (in this CV fold)
    Error_test[k] = np.sum(y_est_test != y_test) / float(len(y_est_test))
    Error_train[k] = np.sum(y_est_train != y_train) / float(len(y_est_train))
    # Number of miss-classifications
    #print('Number of miss-classifications:\n\t {0} out of {1}'.format(np.sum(y_est_test != y_test),len(y_test)))
    k+=1

print('The generalization test error is {0}'.format(np.mean(Error_test)))
print('The generalization training error is {0}'.format(np.mean(Error_train)))


#%% 2-level CV
K1 = 10 #outer
K2 = 10 #inner
lr_lambda = np.logspace(-2, -1, 10)
knn_K = [1, 2]

Error_val_lr = np.empty((K2,len(lr_lambda)))
Error_gen_lr = np.empty(len(lr_lambda))
Error_test_lr = np.empty(K1)
Error_val_knn = np.empty((K2,len(knn_K)))
Error_gen_knn = np.empty(len(knn_K))
Error_test_knn = np.empty(K1)
#Error_val_baseline = np.empty(K2)
Error_test_baseline = np.empty(K1)
opt_lr_lambda = np.empty(K1)
opt_knn_K = np.empty(K1, dtype=int)

for k, (par_index, test_index) in enumerate(model_selection.KFold(n_splits=K1,shuffle=True).split(y)):
    print('Computing outer CV fold: {0}/{1}..'.format(k+1,K1))
    # extract par and test set for current outer CV fold
    X_par, y_par = X[par_index,:], y[par_index]
    X_test, y_test = X[test_index,:], y[test_index]
    K1_ratio = len(y_test)/len(y)
    
    for l, (train_index, val_index) in enumerate(model_selection.KFold(n_splits=K2,shuffle=True).split(y_par)):
        print('-Computing inner CV fold: {0}/{1}..'.format(l+1,K2))
        # extract train and val set for current inner CV fold
        X_train, y_train = X_par[train_index,:], y_par[train_index]
        X_val, y_val = X_par[val_index,:], y_par[val_index]
        K2_ratio = len(y_val)/len(y_par)
        
        X_train_lr, X_val_lr = np.delete(X_train,2,1), np.delete(X_val,2,1)
        for i, lambda_val in enumerate(lr_lambda):
            logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, C=1/lambda_val, max_iter=1000)
            logreg.fit(X_train_lr,y_train)
            y_est_val = logreg.predict(X_val_lr)
            Error_val_lr[l,i] = K2_ratio * np.sum(y_est_val != y_val) / float(len(y_val))
        
        X_train_knn, X_val_knn = X_train[:,[3,4]], X_val[:,[3,4]]
        for i, knn_val in enumerate(knn_K):
            knclassifier = KNeighborsClassifier(n_neighbors=knn_val)
            knclassifier.fit(X_train_knn, y_train)
            y_est_val = knclassifier.predict(X_val_knn)
            Error_val_knn[l,i] = K2_ratio * np.sum(y_est_val != y_val) / float(len(y_val))
        
        #y_est_val = np.bincount(y_train).argmax()
        #Error_val_baseline[l] = np.sum(y_est_val != y_val) / float(len(y_val))
    Error_gen_lr = np.sum(Error_val_lr,0)
    Error_gen_knn = np.sum(Error_val_knn,0)
    
    opt_lr_lambda[k] = lr_lambda[np.argmin(Error_gen_lr)]
    opt_knn_K[k] = knn_K[np.argmin(Error_gen_knn)]
    
    X_par_lr, X_test_lr = np.delete(X_par,2,1), np.delete(X_test,2,1)
    logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, C=1/opt_lr_lambda[k], max_iter=1000)
    logreg.fit(X_par_lr,y_par)
    y_est_test = logreg.predict(X_test_lr)
    Error_test_lr[k] = np.sum(y_est_test != y_test) / float(len(y_test))
    
    X_par_knn, X_test_knn = X_par[:,[3,4]], X_test[:,[3,4]]
    knclassifier = KNeighborsClassifier(n_neighbors=opt_knn_K[k])
    knclassifier.fit(X_par_knn, y_par)
    y_est_test = knclassifier.predict(X_test_knn)
    Error_test_knn[k] = np.sum(y_est_test != y_test) / float(len(y_test))
    
    y_est_test = np.bincount(y_par).argmax()
    Error_test_baseline[k] = np.sum(y_est_test != y_test) / float(len(y_test))

for i in range(K1):
    print('K: {0}| E_knn: {1}| Lambda: {2}| E_lr: {3}| E_base: {4}'.format(opt_knn_K[i], Error_test_knn[i], opt_lr_lambda[i], Error_test_lr[i], Error_test_baseline[i]))

plt.figure()
plt.hist(opt_lr_lambda, bins=10)
plt.xlabel('Lambda')
plt.ylabel('frequency')
plt.show()


#%% Plots
plt.figure(figsize=(20,10))
for i in range(0,M-1):
    plt.subplot(2, 3, i+1)
    plt.plot(X[:,i],y,'.')
plt.show()

plt.figure(figsize=(20,20))
plt.title('Location')
plt.scatter(X[:,4], X[:,3], s=200, c=y, alpha=0.75)
plt.show()

