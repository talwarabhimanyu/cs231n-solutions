import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  ###################### Gradient Computation ################################
  # Li = Sum_over_all_j_not_equal_to_yi Max[0, i_score_j - (i_score_yi - 1)]
  #         where yi is the correct label and j's are labels other than yi
  #         i_score_k = X[i,:]*W[:,k]
  # INTUITION
  # Li can be written = [i_score_j1 - (i_score_yi-1)] + ... 
  #                     + [i_score_jp - (i_score_yi-1)] 
  #     where j1,..,jp are labels such that their margin w.r.t yi is >0.
  # Consider derivative of Li w.r.t dW[n,k]. Two scenarios are possible:
  #     (1) k = yi: Looking at the expression for Li, an increase in W[n,k]
  #         by 1 will decrease Li by p * X[i,n], and so the contribution to
  #         gradient dW[n,k] of Li = -1*p*X[i,n].
  #     (2) k <> yi: An increase of 1 in W[n,k] will increase Li by p*X[i,n]
  #         and so the contribution to dW[n,k] = p*X[i,n].
  #
  ############################################################################ 
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:]
        dW[:,y[i]] -= X[i,:]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  num_class = W.shape[1] 
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  # svm_margin has dimensions N x C                                           #
  #############################################################################
  scores = np.matmul(X, W)
  svm_margin = np.maximum(scores - np.reshape(scores[np.arange(num_train),y],(num_train,-1))+1,0)
  svm_margin[np.arange(num_train),y] = 0
  loss = np.sum(svm_margin)/num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  a = svm_margin
  a[svm_margin > 0] = 1
  a[np.arange(num_train),y] = -1*np.sum(a,axis=1)
  d = dW + np.matmul(X.T,a)/num_train
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  # Add regularization
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  return loss, dW
