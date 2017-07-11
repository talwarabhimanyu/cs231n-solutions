from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape 

    # Compute the forward pass - some terminology below:
    # (1) zi refers to linear combination of inputs going into layer i (a layer is one 
    #     in which neurons have some non-linear activation functions) 
    # (2) hi refers to outputs coming out of layer i
    scores = None
    z1 = np.matmul(X, W1)                                       #Size: NxH
    # Calculate ReLU output: h1
    h1 = np.maximum(0, z1 + b1)                                 #Size: NxH
    z2 = np.matmul(h1, W2)                                      #Size: NxC
    scores = z2

    # Calculate Softmax output: h2
    eW2X2 = np.exp(z2 + b2)                                     #Size: NxC
    h2 = eW2X2/np.sum(eW2X2,axis=1,keepdims=True)               #Size: NxC

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # Assignment requires use of the softmax loss function. 
    #############################################################################
    loss = -np.sum(np.log(h2[np.arange(N),y]))/N + reg*np.sum(np.power(W1,2)) + reg*np.sum(np.power(W2,2))

    # Backward pass: compute gradients
    grads = {}
    
    dL_by_dz2 = -h2                                                 #Size: NxC
    dL_by_dz2[np.arange(N), y] += 1                                 #Size: NxC

    dL_by_dW2 = (-1.0/N)*np.matmul(h1.T, dL_by_dz2) + 2.0*reg*W2    #Size: HxC
    dL_by_db2 = (-1.0/N)*np.sum(dL_by_dz2.T,axis=1)
    
    grads.update({'W2': dL_by_dW2})
    grads.update({'b2': dL_by_db2})

    dL_by_dh1 = np.matmul(dL_by_dz2, W2.T)*(z1 > 0)                 #Size: NxH
    dL_by_dW1 = (-1.0/N)*np.matmul(X.T, dL_by_dh1) + 2.0*reg*W1     #Size: DxH
    dL_by_db1 = (-1.0/N)*np.sum(dL_by_dh1.T, axis=1)                #Size: DxH
    
    grads.update({'W1': dL_by_dW1})
    grads.update({'b1': dL_by_db1})

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      random_idx = np.random.choice(np.arange(num_train), size=batch_size, replace=True)
      X_batch = X[random_idx,:]
      y_batch = y[random_idx]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params.update({'W1':(self.params['W1'] - learning_rate*grads['W1'])})
      self.params.update({'b1':(self.params['b1'] - learning_rate*grads['b1'])})
      self.params.update({'W2':(self.params['W2'] - learning_rate*grads['W2'])})
      self.params.update({'b2':(self.params['b2'] - learning_rate*grads['b2'])})
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    z1 = np.matmul(X, W1)                           #size: NxH
    h1 = np.maximum(0, z1 + b1)                    #Size: NxH
    z2 = np.matmul(h1, W2)                          #Size: NxC
    unnormalized_probabilities = np.exp(z2 + b2)    #Size: NxC
    h2 = unnormalized_probabilities/np.sum(unnormalized_probabilities,axis=1,keepdims=True)   #Size:NxC
    y_pred = np.argmax(h2, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


