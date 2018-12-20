import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    distance = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    distance = distance.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    margin = ax.contour(xx, yy, distance, [-1.0,1.0], colors = 'black')
    return out, margin


def SVM_plot(X0, X1, label, model):
    xx, yy = make_meshgrid(X0, X1)

    fig, sub = plt.subplots()

    plot_contours(sub, model, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)

    sub.scatter(X0, X1, c=label, cmap=plt.cm.coolwarm, s=40, edgecolors='k')
    sub.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], c = 'k', marker ='x')
    sub.set_xlim(xx.min(), xx.max())
    sub.set_ylim(yy.min(), yy.max())

    plt.show()