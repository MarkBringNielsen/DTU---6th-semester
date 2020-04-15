# exercise Load data
import numpy as np
import pandas as pd

from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show, hist, ylim)
from scipy.stats import zscore
from scipy.linalg import svd
import matplotlib.pyplot as plt
import torch

from matplotlib.pylab import (semilogx, loglog, 
                           title, grid)


import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from toolbox_02450 import train_neural_net, draw_neural_net

import scipy.stats as st

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

from sklearn.neighbors import KNeighborsClassifier


#######################################################
### Load the Wine csv data using the Pandas library ###
#######################################################
filename = 'wine.csv'
df = pd.read_csv(filename, header=None)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df.values

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(1, 14)  
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
#attributeNames = np.asarray(df.columns[cols])
#print(attributeNames)

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
classLabels = raw_data[:,0] # 0 takes the first column
# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels)
# We can assign each type of WIne class with a number by making a
# Python dictionary as so:
classDict = dict(zip(classNames,range(len(classNames))))
# The function zip simply "zips" togetter the classNames with an integer,
# like a zipper on a jacket. 
# For instance, you could zip a list ['A', 'B', 'C'] with ['D', 'E', 'F'] to
# get the pairs ('A','D'), ('B', 'E'), and ('C', 'F'). 
# A Python dictionary is a data object that stores pairs of a key with a value. 
# This means that when you call a dictionary with a given key, you 
# get the stored corresponding value. Try highlighting classDict and press F9.
# You'll see that the first (key, value)-pair is ('Iris-setosa', 0). 
# If you look up in the dictionary classDict with the value 'Iris-setosa', 
# you will get the value 0. Try it with classDict['Iris-setosa']

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])
# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of 
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the 
# list to a numpy array". 
# Try running this to get a feel for the operation: 
# list = [0,1,2]
# new_list = [element+10 for element in list]



#####################
#Choose the regularisation problem lige this: Taken from 7_2_1. 
#X,y = X[:,:10], X[:,10:]
y = X[:,0] #Alcohol
X = X[:, 1:] #[2,4,9,10,12]]
# Choose Alcohol = offset, Ash, Magnesium, Color Intensity, Hue, Proline. (1 = 0 + 3 + 5 + 10 + 11 + 13) (/Numbers are WITH offset)
#####################

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames)


# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
M = M+1


#attributeNames = ('Offset', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline')
#attributeNames = ('Offset', 'Ash', 'Magnesium', 'Color intensity', 'Hue' 'Proline')
attributeNames = ('Offset', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline')

#Standalize the data to zero mean and standard deviation of 1
X_standarized = zscore(X, ddof=1) #Do the standalization 



# Parameters for neural network classifier
n_hidden_units_m1 = 1      # number of hidden units
n_hidden_units_m2 = 30      # number of hidden units
n_hidden_units_m3 = 60      # number of hidden units
n_hidden_units_m4 = 90
n_replicates = 3        # number of networks trained in each k-fold
max_iter = 10000
number_of_ANN_modelse = 4;
hidden_units_array = np.empty([number_of_ANN_modelse,1])
hidden_units_array[0] = n_hidden_units_m1
hidden_units_array[1] = n_hidden_units_m2
hidden_units_array[2] = n_hidden_units_m3
hidden_units_array[3] = n_hidden_units_m4


# Define the models
ANN_model_1 = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units_m1), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units_m1, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn_1 = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

ANN_model_2 = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units_m2), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units_m2, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn_2 = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

ANN_model_3 = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units_m3), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units_m3, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn_3 = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

ANN_model_4 = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units_m4), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units_m4, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn_4 = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

#ANN_error_1 = [] # make a list for storing generalizaition error in each loop
#ANN_error_2 = [] # make a list for storing generalizaition error in each loop
#ANN_error_3 = [] # make a list for storing generalizaition error in each loop

############################################
#Use exercise 8.1.1
###########################################


## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = 5
K2 = 5
CV = model_selection.KFold(K1, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
#lambdas = np.power(10.,range(-5,9))
lambdas = [0.0001,0.001, 0.01,0.1,1,10,20,30,40,100,1000,10000]
#lambdas = np.power(10.,np.arange(-5,9,0.3))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K2,K1))
Error_test = np.empty((K2,K1))
Error_train_rlr = np.empty((K2,K1))
Error_test_rlr = np.empty((K2,K1))
Error_train_nofeatures = np.empty((K2,K1))
Error_test_nofeatures = np.empty((K2,K1))
w_rlr = np.empty((M,K1))
#mu = np.empty((K1, M-1))
#sigma = np.empty((K1, M-1))
#w_noreg = np.empty((M,K1))


ANN_error = np.empty([K2,number_of_ANN_modelse]) #10 is for crossvaidation and 3 is for ANN models
ANN_best_error = np.empty([K2,1])
optimal_h_array = np.empty((K2,1))


optimal_lambda_array = np.empty((K2))
dummy_optimal_lambda_array = np.empty((K2))

Error_test_ANN = np.empty([K1])
Error_test_lin_reg = np.empty([K1])
Error_test_baseline = np.empty([K1])

#For statistics
CI_ab = np.empty([K1,2]) 
p_ab = np.empty([K1,1])
CI_ac = np.empty([K1,2]) 
p_ac = np.empty([K1,1])
CI_bc = np.empty([K1,2]) 
p_bc = np.empty([K1,1])

k=0
iii = 0
for train_index_outer, test_index_outer in CV.split(X,y):
    print('iii is:')
    print(iii) 
    # extract training and test set for current CV fold
    X_train_outer = X[train_index_outer]
    y_train_outer = y[train_index_outer]
    X_test_outer = X[test_index_outer]
    y_test_outer = y[test_index_outer]
        
    for train_index_inner, test_index_inner in CV.split(X_train_outer,y_train_outer):
        
        # extract training and test set for current CV fold
        X_train_inner = X[train_index_inner]
        y_train_inner = y[train_index_inner]
        X_test_inner = X[test_index_inner]
        y_test_inner = y[test_index_inner]
        internal_cross_validation = 10    
        
        ######## "The s for loop" where each model is trained
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train_inner, y_train_inner, lambdas, internal_cross_validation)
        
            # Extract training and test set for current CV fold, convert to tensors
        X_train_ANN = torch.Tensor(X_train_inner) # X[train_index,:])
        y_train_ANN = torch.Tensor(y_train_inner) # y[train_index])
        X_test_ANN = torch.Tensor(X_test_inner) # X[test_index,:])
        y_test_ANN = torch.Tensor(y_test_inner) # y[test_index])
        
        ###################
        ### ANN model 1 ###
        ###################
        # Train the net on training data
        net_m1, final_loss, learning_curve = train_neural_net(ANN_model_1,
                                                           loss_fn_1,
                                                           X=X_train_ANN,
                                                           y=y_train_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M1 Best loss: {}\n'.format(final_loss))
        
        # Determine estimated class labels for test set        
        y_test_est_m1 = net_m1(X_test_ANN).detach().numpy() ### 
        # Determine Mean square error      
        mse = np.square(y_test_inner-np.squeeze(y_test_est_m1)).sum(axis=0)/y_test_inner.shape[0]
        # Save it
        ANN_error[k,0] = mse  # np.asarray(mse)
       
        
        ### ANN model 2
        # Train the net on training data
        net_m2, final_loss, learning_curve = train_neural_net(ANN_model_2,
                                                           loss_fn_2,
                                                           X=X_train_ANN,
                                                           y=y_train_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M2 Best loss: {}\n'.format(final_loss))
        
       # Determine estimated class labels for test set        
        y_test_est_m2 = net_m2(X_test_ANN).detach().numpy() ### 
        # Determine Mean square error      
        mse = np.square(y_test_inner-np.squeeze(y_test_est_m2)).sum(axis=0)/y_test_inner.shape[0]
        # Save it
        ANN_error[k,1] = mse  # np.asarray(mse)
       
        
        ### ANN model 3
        # Train the net on training data
        net_m3, final_loss, learning_curve = train_neural_net(ANN_model_3,
                                                           loss_fn_3,
                                                           X=X_train_ANN,
                                                           y=y_train_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M3 Best loss: {}\n'.format(final_loss))
         # Determine estimated class labels for test set        
        y_test_est_m3 = net_m3(X_test_ANN).detach().numpy() ### 
        # Determine Mean square error      
        mse = np.square(y_test_inner-np.squeeze(y_test_est_m3)).sum(axis=0)/y_test_inner.shape[0]
        # Save it
        ANN_error[k,2] = mse  # np.asarray(mse)
       
        ### ANN model 4
        # Train the net on training data
        net_m4, final_loss, learning_curve = train_neural_net(ANN_model_4,
                                                           loss_fn_4,
                                                           X=X_train_ANN,
                                                           y=y_train_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M4 Best loss: {}\n'.format(final_loss))
         # Determine estimated class labels for test set        
        y_test_est_m4 = net_m4(X_test_ANN).detach().numpy() ### 
        # Determine Mean square error      
        mse = np.square(y_test_inner-np.squeeze(y_test_est_m4)).sum(axis=0)/y_test_inner.shape[0]
        # Save it
        ANN_error[k,3] = mse  # np.asarray(mse)
        
        
        
        # Standardize outer fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        #Uncommented these 4 lines (TK)
        #mu[k, :] = np.mean(X_train[:, 1:], 0)
        #sigma[k, :] = np.std(X_train[:, 1:], 0)
        
        #X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
        #X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
        
        Xty = X_train_inner.T @ y_train_inner
        XtX = X_train_inner.T @ X_train_inner
        
        # Compute mean squared error without using the input data at all - The base line model
        Error_train_nofeatures[k] = np.square(y_train_inner-y_train_inner.mean()).sum(axis=0)/y_train_inner.shape[0]
        Error_test_nofeatures[k] = np.square(y_test_inner-y_test_inner.mean()).sum(axis=0)/y_test_inner.shape[0]
    
        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k,iii] = np.square(y_train_inner-X_train_inner @ w_rlr[:,k]).sum(axis=0)/y_train_inner.shape[0]
        Error_test_rlr[k,iii] = np.square(y_test_inner-X_test_inner @ w_rlr[:,k]).sum(axis=0)/y_test_inner.shape[0]
        
        dummy_optimal_lambda_array[k] = opt_lambda
        
        
    
        
    
        k+=1
    
    #TK:
    
    #Pick out the best ANN model
    index_dummy = np.where(ANN_error == np.amin(ANN_error)) #Minimun index, to safe h*
    optimal_h_array[iii] = index_dummy[1]
    
    
        
    # Extract training and test set for current CV fold, convert to tensors
    X_train_outer_ANN = torch.Tensor(X_train_outer) # X[train_index,:])
    y_train_outer_ANN = torch.Tensor(y_train_outer) # y[train_index])
    X_test_outer_ANN = torch.Tensor(X_test_outer) # X[test_index,:])
    y_test_outer_ANN = torch.Tensor(y_test_outer) # y[test_index])

#Train the best model on the training set -- this could have been done in a nicer way... But python dont have switch case and I dont want to make a model array :p
    if index_dummy[1] == 0:
        net_m_final, final_loss, learning_curve = train_neural_net(ANN_model_1,
                                                           loss_fn_1,
                                                           X=X_train_outer_ANN,
                                                           y=y_train_outer_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M1 Best loss: {}\n'.format(final_loss))
    elif index_dummy[1] == 1:
        net_m_final, final_loss, learning_curve = train_neural_net(ANN_model_2,
                                                           loss_fn_2,
                                                           X=X_train_outer_ANN,
                                                           y=y_train_outer_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M2 Best loss: {}\n'.format(final_loss))
    elif index_dummy[1] == 2:
        net_m_final, final_loss, learning_curve = train_neural_net(ANN_model_3,
                                                           loss_fn_3,
                                                           X=X_train_outer_ANN,
                                                           y=y_train_outer_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M3 Best loss: {}\n'.format(final_loss))
    elif index_dummy[1] == 3:
        net_m_final, final_loss, learning_curve = train_neural_net(ANN_model_4,
                                                           loss_fn_4,
                                                           X=X_train_outer_ANN,
                                                           y=y_train_outer_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M4 Best loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set        
    y_test_est_m_final = net_m_final(X_test_outer_ANN).detach().numpy() ### 
    # Determine Mean square error      
    mse = np.square(y_test_outer-np.squeeze(y_test_est_m_final)).sum(axis=0)/y_test_outer.shape[0]
    #Save the error
    Error_test_ANN[iii] = mse #Minimum value 
    
    #### Linear regression model
    #Pick out the best optimal lambda value
    min_lambda_index = np.where(Error_test_rlr[:,iii] == np.amin(Error_test_rlr[:,iii]))
    optimal_lambda_array[iii] = dummy_optimal_lambda_array[min_lambda_index]
    #Train the model:
    Xty = X_train_outer.T @ y_train_outer
    XtX = X_train_outer.T @ X_train_outer

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = optimal_lambda_array[iii] * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr_o = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    #Error_train_rlr[k,iii] = np.square(y_train_outer-X_train_outer@ w_rlr_o).sum(axis=0)/y_train_outer.shape[0]
    Error_test_lin_reg[iii] = np.square(y_test_outer-X_test_outer @ w_rlr_o).sum(axis=0)/y_test_outer.shape[0]
    
    #### Base line model
     # Compute mean squared error - The base line model
    Error_test_baseline[iii] = np.square(y_test_outer-y_train_outer.mean()).sum(axis=0)/y_test_outer.shape[0]
    
    #Do the statistic evaluation. A is linear regression, B is ANN and C is baseline
    yhatA = X_test_outer @ w_rlr_o
    zA = np.abs(y_test_outer - yhatA ) ** 2
    # yhatB = y_test_est_m_final
    zB = np.abs(y_test_outer - np.squeeze(y_test_est_m_final) ) ** 2
    # yhatC = y_train_outer.mean()
    zC = np.abs(y_test_outer - y_train_outer.mean() ) ** 2
    
    alpha = 0.05
    #Compare linear regression with ANN
    z_ab = zA - zB
    CI_ab[iii,:] = st.t.interval(1-alpha, len(z_ab)-1, loc=np.mean(z_ab), scale=st.sem(z_ab))  # Confidence interval
    p_ab[iii] = st.t.cdf( -np.abs( np.mean(z_ab) )/st.sem(z_ab), df=len(z_ab)-1)  # p-value
    #Compare linear regression with base line
    z_ac = zA - zC
    CI_ac[iii,:] = st.t.interval(1-alpha, len(z_ac)-1, loc=np.mean(z_ac), scale=st.sem(z_ac))  # Confidence interval
    p_ac[iii] = st.t.cdf( -np.abs( np.mean(z_ac) )/st.sem(z_ac), df=len(z_ac)-1)  # p-value
    #Compare ANN with base line
    z_bc = zB - zC
    CI_bc[iii,:] = st.t.interval(1-alpha, len(z_bc)-1, loc=np.mean(z_bc), scale=st.sem(z_bc))  # Confidence interval
    p_ac[iii] = st.t.cdf( -np.abs( np.mean(z_bc) )/st.sem(z_bc), df=len(z_bc)-1)  # p-value
  #  Error_test_ANN[iii]
   # Error_test_lin_reg[iii]
   # Error_test_baseline[iii]
    
    iii+=1
    k = 0
    
    
        # Estimate weights for unregularized linear regression, on entire training set
        #w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
        # Compute mean squared error without regularization
        #Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
        #Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
        # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
        #m = lm.LinearRegression().fit(X_train, y_train)
        #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]
        
        
#print(Error_test_rlr)        
        


print('Linear regression - Test error')
#print(np.mean(Error_test_rlr,axis=0))
print(Error_test_lin_reg)
print('Optimal Lambda values)')
print(optimal_lambda_array)   
print('')
print('Baseline')     
print(Error_test_baseline)
#print(np.mean(Error_test_nofeatures,axis=0))

print('')
print('ANN test error')     
#print(ANN_best_error)
print(Error_test_ANN)
print('Optimal h values)')
print(optimal_h_array)   

print('Statistics')
print('Compare linear regression with ANN')
print(CI_ab)
print(p_ab)
print('Compare linear regression with baseline')
print(CI_ac)
print(p_ac)
print('Compare ANN with baseline')
print(CI_bc)
print(p_bc)


#Classification

#for outer_train_set, outer_valid_set in train_set(K_fold):
#    for inner_train_set, inner_valid_set in outer_train_set(K_fold):
#        Train model and find the optimal hyper-parameters 
#    Train model using the optimal hyper-parameters found on the inner loop
#    Evaluate performance on outer_valid_set 

X = df.drop(0, axis=1)
y = df[0]

## Crossvalidation
# Create crossvalidation partition for evaluation
K1 = 10
K2 = 10
CV = model_selection.KFold(K1, shuffle=True)


k=0
iii = 0
for train_index_outer, test_index_outer in CV.split(X,y):
    print('iii is:')
    print(iii) 
        
    k_scores = []
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = model_selection.cross_val_score(knn, X, y, cv=CV, scoring='accuracy')
        k_scores.append(scores.mean())
        
    print(k_scores.index(max(k_scores)), max(k_scores), ' : ', k_scores)
    
    #TK:
    
    #Pick out the best ANN model
    index_dummy = np.where(ANN_error == np.amin(ANN_error)) #Minimun index, to safe h*
    optimal_h_array[iii] = index_dummy[1]
    
    
        
    # Extract training and test set for current CV fold, convert to tensors
    X_train_outer_ANN = torch.Tensor(X_train_outer) # X[train_index,:])
    y_train_outer_ANN = torch.Tensor(y_train_outer) # y[train_index])
    X_test_outer_ANN = torch.Tensor(X_test_outer) # X[test_index,:])
    y_test_outer_ANN = torch.Tensor(y_test_outer) # y[test_index])

#Train the best model on the training set -- this could have been done in a nicer way... But python dont have switch case and I dont want to make a model array :p
    if index_dummy[1] == 0:
        net_m_final, final_loss, learning_curve = train_neural_net(ANN_model_1,
                                                           loss_fn_1,
                                                           X=X_train_outer_ANN,
                                                           y=y_train_outer_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M1 Best loss: {}\n'.format(final_loss))
    elif index_dummy[1] == 1:
        net_m_final, final_loss, learning_curve = train_neural_net(ANN_model_2,
                                                           loss_fn_2,
                                                           X=X_train_outer_ANN,
                                                           y=y_train_outer_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M2 Best loss: {}\n'.format(final_loss))
    elif index_dummy[1] == 2:
        net_m_final, final_loss, learning_curve = train_neural_net(ANN_model_3,
                                                           loss_fn_3,
                                                           X=X_train_outer_ANN,
                                                           y=y_train_outer_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M3 Best loss: {}\n'.format(final_loss))
    elif index_dummy[1] == 3:
        net_m_final, final_loss, learning_curve = train_neural_net(ANN_model_4,
                                                           loss_fn_4,
                                                           X=X_train_outer_ANN,
                                                           y=y_train_outer_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        print('\n\t M4 Best loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set        
    y_test_est_m_final = net_m_final(X_test_outer_ANN).detach().numpy() ### 
    # Determine Mean square error      
    mse = np.square(y_test_outer-np.squeeze(y_test_est_m_final)).sum(axis=0)/y_test_outer.shape[0]
    #Save the error
    Error_test_ANN[iii] = mse #Minimum value 
    
    #### Linear regression model
    #Pick out the best optimal lambda value
    min_lambda_index = np.where(Error_test_rlr[:,iii] == np.amin(Error_test_rlr[:,iii]))
    optimal_lambda_array[iii] = dummy_optimal_lambda_array[min_lambda_index]
    #Train the model:
    Xty = X_train_outer.T @ y_train_outer
    XtX = X_train_outer.T @ X_train_outer

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = optimal_lambda_array[iii] * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr_o = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    #Error_train_rlr[k,iii] = np.square(y_train_outer-X_train_outer@ w_rlr_o).sum(axis=0)/y_train_outer.shape[0]
    Error_test_lin_reg[iii] = np.square(y_test_outer-X_test_outer @ w_rlr_o).sum(axis=0)/y_test_outer.shape[0]
    
    #### Base line model
     # Compute mean squared error - The base line model
    Error_test_baseline[iii] = np.square(y_test_outer-y_train_outer.mean()).sum(axis=0)/y_test_outer.shape[0]
    
    #Do the statistic evaluation. A is linear regression, B is ANN and C is baseline
    yhatA = X_test_outer @ w_rlr_o
    zA = np.abs(y_test_outer - yhatA ) ** 2
    # yhatB = y_test_est_m_final
    zB = np.abs(y_test_outer - np.squeeze(y_test_est_m_final) ) ** 2
    # yhatC = y_train_outer.mean()
    zC = np.abs(y_test_outer - y_train_outer.mean() ) ** 2
    
    alpha = 0.05
    #Compare linear regression with ANN
    z_ab = zA - zB
    CI_ab[iii,:] = st.t.interval(1-alpha, len(z_ab)-1, loc=np.mean(z_ab), scale=st.sem(z_ab))  # Confidence interval
    p_ab[iii] = st.t.cdf( -np.abs( np.mean(z_ab) )/st.sem(z_ab), df=len(z_ab)-1)  # p-value
    #Compare linear regression with base line
    z_ac = zA - zC
    CI_ac[iii,:] = st.t.interval(1-alpha, len(z_ac)-1, loc=np.mean(z_ac), scale=st.sem(z_ac))  # Confidence interval
    p_ac[iii] = st.t.cdf( -np.abs( np.mean(z_ac) )/st.sem(z_ac), df=len(z_ac)-1)  # p-value
    #Compare ANN with base line
    z_bc = zB - zC
    CI_bc[iii,:] = st.t.interval(1-alpha, len(z_bc)-1, loc=np.mean(z_bc), scale=st.sem(z_bc))  # Confidence interval
    p_ac[iii] = st.t.cdf( -np.abs( np.mean(z_bc) )/st.sem(z_bc), df=len(z_bc)-1)  # p-value
  #  Error_test_ANN[iii]
   # Error_test_lin_reg[iii]
   # Error_test_baseline[iii]
    
    iii+=1
    k = 0
    



print('Linear regression - Test error')
#print(np.mean(Error_test_rlr,axis=0))
print(Error_test_lin_reg)
print('Optimal Lambda values)')
print(optimal_lambda_array)   
print('')
print('Baseline')     
print(Error_test_baseline)
#print(np.mean(Error_test_nofeatures,axis=0))

print('')
print('ANN test error')     
#print(ANN_best_error)
print(Error_test_ANN)
print('Optimal h values)')
print(optimal_h_array)   

print('Statistics')
print('Compare linear regression with ANN')
print(CI_ab)
print(p_ab)
print('Compare linear regression with baseline')
print(CI_ac)
print(p_ac)
print('Compare ANN with baseline')
print(CI_bc)
print(p_bc)