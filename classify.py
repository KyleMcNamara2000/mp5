# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set

return - a list containing predicted labels for dev_set
"""

import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    W = np.array([0.0] * 32 * 32 * 3)
    #W = [0] * 32 * 32 * 3
    b = 0
    for ja in range(max_iter):
        for i in range(len(train_set)):
            dot = np.sign(np.dot(W, train_set[i]))
            s = 1
            if train_labels[i] is np.bool_(False):
                s = -1
            if dot != s:
                k = learning_rate * s
                dodo = np.dot(k, train_set[i])
                W += dodo
                b += learning_rate
                #print(b)



    return W, b


    '''
    oldW = np.array([0] * 32 * 32 * 3)
    #oldW = [0] * 32 * 32 * 3
    b = 0 #b = w0 * x0, x0 = 1
    for i in range(len(train_set)):
        for j in range(len(train_set[i])):
            #oldW[i][j], train_set[i][j]
            x = train_set[i][j]
            wi = oldW[j]
            y_correct = train_labels[i]
            y_pred = x * wi
            if y_pred != y_correct:
                oldW[j] = wi + learning_rate * (y_correct - y_pred) * x
                b -= learning_rate
            else:
                b += learning_rate
    return oldW, b
    '''

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    guesses = []
    #dot product W .
    for i in range(len(dev_set)):
        sum = np.dot(W, dev_set[i])
        sum += b
        guesses.append(np.sign(sum))
    return guesses




def correctLabel(L):
    negs = 0
    pos = 0
    for x in L:
        if x[1] == 1:
            pos += 1
        else:
            negs += 1
    if pos > negs:
        return 1
    else:
        return -1

def classifyKNN(train_set, train_labels, dev_set, k):
    # TODO: Write your code here
    return []
    '''
    out = []
    for i in range(len(dev_set)):
        nearest = [] #nearest pairs (distance, trained value)
        for j in range(len(train_set)):
            curr_dist = np.linalg.norm(dev_set[i], zip(train_set[j])[0])
            '''
            for m in range(32 * 32 * 3):
                curr_dist += np.abs(dev_set[i][m] - train_set[j][m])
            '''

            if j < k:
                nearest.append((curr_dist, train_labels[j]))
                nearest.sort()
            elif curr_dist < nearest[-1][0]:
                nearest.pop(-1)
                nearest.append((curr_dist, train_labels[j]))
                nearest.sort()
        out.append(correctLabel(nearest))

    return out
    '''
