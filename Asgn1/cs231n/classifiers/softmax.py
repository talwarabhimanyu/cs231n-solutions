import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  ########################### Softmax Loss & Gradient #########################
  # Contribution to loss of the ith training example is:
  #     Li = -log(exp(X[i,:]*W[:,yi])) / (Sum_over_all_js exp(X[i,:]*W[:,j]))
  #############################################################################
  num_train = X.shape[0]
  num_features = X.shape[1]
  num_classes = W.shape[1]
  for iIter in range(0,num_train):
      prob_sum = np.zeros(num_classes)
      true_y = y[iIter]
      for jIter in range(0,num_classes):
          prob_sum[jIter] = np.exp(np.matmul(X[iIter,:], W[:,jIter]))
      dW += np.matmul(np.transpose(np.reshape(X[iIter,:],(1,num_features))), np.reshape(prob_sum,(1,num_classes)))/np.sum(prob_sum)
      dW[:,true_y] -= np.transpose(X[iIter,:])
      loss += -np.log(prob_sum[true_y]/np.sum(prob_sum))
  
  # Add regularization terms
  loss = loss/num_train + reg*np.sum(np.power(W,2))
  dW = dW/num_train + 2*reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  num_features = W.shape[0]
  num_classes = W.shape[1]

  f = np.exp(np.matmul(X, W))
  f_sum = np.reshape(np.sum(f, axis=1),(num_train,1))
  loss = np.sum(-np.log(np.reshape(f[np.arange(num_train),y],(num_train,-1)) / f_sum))/num_train + reg*np.sum(np.power(W,2))
  dW = np.matmul(np.transpose(X), f / f_sum)
  nc_temp = np.zeros((num_train, num_classes))
  nc_temp[np.arange(num_train),y] = 1

  dW -= np.transpose(np.matmul(np.transpose(nc_temp), X))
  dW = dW/num_train
  dW += 2*reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

