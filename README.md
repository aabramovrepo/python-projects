# python-projects

This repository contains a bunch of some useful algorithms implemented in Python. More details about the algorithms and their implementations you can find following the links below.

## Kd_Tree

A Kd-tree, or K-dimensional tree, is a generalization of a binary search tree that stores points in a k-dimensional space. In computer science it is often used for organizing some number of points in a space with k dimensions. Kd-trees are very useful for range and nearest neighbor (NN) searches, it is a very common operation in computer vision, computational geometry, data mining, machine learning.

The current project contains functions for building 2- and 3-dimensional Kd trees out of given point sets in 2- and 3-dimensional Cartesian space. 

* Kd_Tree.py -- 2-dimensional Kd tree
* Kd_Tree_3D.py -- 3-dimensional Kd tree

More details about building Kd trees ans using them for Nearest Neighbor (NN) search you can find folloiwing the link:

https://salzis.wordpress.com/2014/06/28/kd-tree-and-nearest-neighbor-nn-search-2d-case/

## RANSAC

RANSAC or “RANdom SAmple Consensus” is an iterative method to estimate parameters of a mathematical model from a set of observed data which contains outliers. It is one of classical techniques in computer vision. The current script contains robust linear model estimation using RANSAC. More details about RANSAC and the current implementation you can find following the link:

https://salzis.wordpress.com/2014/06/10/robust-linear-model-estimation-using-ransac-python-implementation/

## RANSAC_SciPy

RANSAC implementation from the Scipy Cookbook for finding a linear model:

http://scipy.github.io/old-wiki/pages/Cookbook/RANSAC

Hovewer, it is actually not the original RANSAC. Looking on that page at the figure above we see a nicely fitted model describing the data with outliers. But going through the implementation you will see that it judges the model using the squared error between the model and testing points. According to this, testing points satisfying the model are added to a set of previously selected points (considered for the modeling) and a new model is estimated using all these points altogether. The final model is tested against all selected points using the squared error. It is not a bad idea, but there are two problems. First, it is not an initial RANSAC algorithm which judges the model by calculating distances from testing points to the model and classifies those ones as outliers whose distances are higher than a pre-defined threshold. Second, the present implementation contains some inconsistencies classifying points being far away from the model as inliers (see points marked with blue crosses in the figure below).

## convex_hull

https://salzis.wordpress.com/2014/05/01/convex-hull-how-to-tell-whether-a-point-is-inside-or-outside/

## particle_filter

https://salzis.wordpress.com/2015/05/25/particle-filters-with-python/

