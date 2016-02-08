#
# The current code implements the RANSAC algorithm for finding a linear model
# It uses the linear least squares for model guessing on random point sets
#
# The current implementation is based on the Cookbook from SciPy: http://wiki.scipy.org/Cookbook/RANSAC
# and the pseudo-code from: http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182
#


import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import math


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """ fit model parameters to data using the RANSAC algorithm
    :param data       a set of observed data points
    :param model      a model that can be fitted to data points
    :param n          the minimum number of data values required to fit the model
    :param k          the maximum number of iterations allowed in the algorithm
    :param t          a threshold value for determining when a data point fits a model
    :param d          the number of close data values required to assert that a model fits well to data
    :param debug      debug information is on if True, off otherwise
    :param return_all
    :return model parameters which best fit the data (or nil if no good model is found)
    """

    print 't = ', t

    iterations = 0
    best_fit = None
    best_err = np.inf
    best_inlier_indices = None

    while iterations < k:

        # pick n randomly selected values from data for modelling
        # the rest will be used for testing the model
        maybe_indices, test_indices = random_partition(n,data.shape[0])

        maybe_inliers = data[maybe_indices,:]
        test_points = data[test_indices]

        # find a model for selected points
        maybe_model = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybe_model)

        #print 'test_err = ', test_err

        # points fitting data with an error smaller than t are considered as inliers
        also_indices = test_indices[test_err < t]
        also_inliers = data[also_indices,:]

        if debug:

            print 'test_err.min()', test_err.min()
            print 'test_err.max()', test_err.max()
            print 'p.mean(test_err)', np.mean(test_err)
            print 'iteration %d:len(also_inliers) = %d' % (iterations,len(also_inliers))

        # check whether there are enough points described by the found model
        if len(also_inliers) > d:

            # concatenate initially selected points and points considered as inliers from the testing set
            better_data = np.concatenate( (maybe_inliers, also_inliers) )

            # perform model fitting on the extended data set
            better_model = model.fit(better_data)
            better_errs = model.get_error(better_data, better_model)
            err = np.mean( better_errs )

            # take this in model in case fitting error decreases
            if err < best_err:

                best_fit = better_model
                best_err = err
                best_inlier_indices = np.concatenate( (maybe_indices, also_indices) )

        iterations += 1

    if best_fit is None:
        raise ValueError("did not meet fit acceptance criteria")

    if return_all:
        return best_fit, best_inlier_indices
    else:
        return best_fit


def ransac_modified(data, model, n, k, t, d):
    """ fit model parameters to data using the RANSAC algorithm
    :param data       a set of observed data points
    :param n          the minimum number of data values required to fit the model
    :param k          the maximum number of iterations allowed in the algorithm
    :param t          a threshold value for determining when a data point fits a model
    :param d          the number of close data values required to assert that a model fits well to data
    :return model parameters which best fit the data (or nil if no good model is found)
    """

    iterations = 0
    best_fit = None
    best_err = np.inf
    best_inlier_indices = None

    while iterations < k:

        # pick n randomly selected values from data for modelling
        # the rest will be used for testing the model
        maybe_indices, test_indices = random_partition(n,data.shape[0])

        maybe_inliers = data[maybe_indices,:]
        test_points = data[test_indices]

        # find a model for selected points
        #maybe_model = model.fit(maybe_inliers)

        # find a line model for the selected points
        x = maybe_inliers[:,0]
        y = maybe_inliers[:,1]

        a = np.vstack([x, np.ones(len(x))]).T
        maybe_model, c = np.linalg.lstsq(a, y)[0]

        # WARNING!!! There is an error here !!!
        #test_err = model.get_error(test_points, maybe_model)
        #test_err = model.get_error_ransac(test_points, maybe_inliers, maybe_model, c)
        test_err = model.get_error_ransac(test_points, maybe_inliers, maybe_model, c)

        print 'test_err = ', test_err

        # points fitting data with an error smaller than t are considered as inliers
        also_indices = test_indices[test_err < 20]
        also_inliers = data[also_indices,:]

        for ind in range(test_points.shape[0]):
            print 'test_point x = ', test_points[ind][0]
            print 'test_point y = ', test_points[ind][1]
            print 'err = ', test_err[ind]
            print ''
            print ''


        plt.figure("Ransac_SciPy")
        plt.plot(data[:,0], data[:,1], marker='o', color='#ff8000', linestyle='None', alpha=0.3)
        plt.plot(maybe_inliers[:,0], maybe_inliers[:,1], marker='o', color='#0080ff', linestyle='None')

        x[0] = -4
        x[1] = 24
        plt.plot(x, maybe_model*x + c, 'r', label='Fitted line', color='#0080ff')

        plt.plot(also_inliers[:,0], also_inliers[:,1], marker='o', color='#ff0000', linestyle='None', alpha=0.5)

        plt.savefig("output/figure_" + str(iterations) + ".png")
        plt.close()


        # check whether there are enough points described by the found model
        if len(also_inliers) > d:

            # concatenate initially selected points and points considered as inliers from the testing set
            better_data = np.concatenate( (maybe_inliers, also_inliers) )

            # perform model fitting on the extended data set
            better_model = model.fit(better_data)
            better_errs = model.get_error(better_data, better_model)
            err = np.mean( better_errs )

            # take this in model in case fitting error decreases
            if err < best_err:

                best_fit = better_model
                best_err = err
                best_inlier_indices = np.concatenate( (maybe_indices, also_indices) )

        iterations += 1

    if best_fit is None:
        raise ValueError("did not meet fit acceptance criteria")

    return best_fit, best_inlier_indices


def random_partition(n, n_data):
    """ Shuffle randomly input data and divide it into two groups with sizes (n) and (len(data) - n) """

    all_indices = np.arange(n_data)
    np.random.shuffle(all_indices)
    indices_1 = all_indices[:n]
    indices_2 = all_indices[n:]

    return indices_1, indices_2


class LinearLeastSquaresModel:
    """ linear system solved using linear least squares """


    def __init__(self, input_columns, output_columns, debug=False):

        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug


    def fit(self, data):
        """ Perform the linear least squares model fitting on the given input
         :param data data set for model fitting
        """

        x = np.vstack([data[:,self.input_columns[0]]]).T
        y = np.vstack([data[:,self.output_columns[0]]]).T

        linear_fit, residuals, rank, s = scipy.linalg.lstsq(x,y)

        return linear_fit


    def get_error(self, data, model):
        """ Compute model fitting error
         :param data set for model testing
         :param model to be tested
         :return
        """

        x = np.vstack([data[:,self.input_columns[0]]]).T
        y = np.vstack([data[:,self.output_columns[0]]]).T

        y_fit = scipy.dot(x, model)
        error = np.sum((y - y_fit)**2, axis=1)  # sum squared error per row

        #error1 = np.array( np.sqrt((y - y_fit)**2) )  # sum squared error per row
        #error1 = np.array( (y - y_fit)**2 )  # sum squared error per row

        error1 = np.array( abs(y - y_fit) )

        error1 = np.reshape(error1, error.shape)
        #print 'error1 = ', error1

        print 'error1 shape = ', error1.shape

        return error1


    def get_error_ransac(self, data, maybe_inliers, model, c):

        #print 'data = ', data
        #print 'maybe_inliers = ', maybe_inliers
        #print 'model = ', model
        #print 'c = ', c

        x = np.vstack([data[:,self.input_columns[0]]]).T
        y = np.vstack([data[:,self.output_columns[0]]]).T

        error = np.zeros(shape=x.shape)

        #start_x = -4  #maybe_inliers[0,0]
        stop_x = 24   #maybe_inliers[1,0]

        #if start_x > stop_x:
        #    start_x = maybe_inliers[1,0]
        #    stop_x = maybe_inliers[0,0]

        #print 'start_x = ', start_x
        #print 'stop_x = ', stop_x

        for ind in range(x.shape[0]):

            cur_x = -4
            error[ind] = 1e6

            print 'point x = ', x[ind]
            print 'point y = ', y[ind]

            #for cur_x in range(start_x, stop_x, 1):
            while cur_x < stop_x:

                cur_y = cur_x*model + c
                dist = math.sqrt((x[ind] - cur_x) ** 2 + (y[ind] - cur_y) ** 2)

                print 'cur_x = ', cur_x
                print 'cur_y = ', cur_y
                print 'dist = ', dist

                if dist < error[ind]:
                    error[ind] = dist

                cur_x += 0.1

            print 'error y = ', error[ind]
            print ''
            print ''

        error = np.reshape(error, (x.shape[0],))

        return error


def fit_line():

    x = np.array([-1, 3])
    y = np.array([-4, 4])

    x1 = np.array([9.9])
    y1 = np.array([153.6])

    m = (y[1] - y[0]) / (x[1] - x[0])
    c = y[1] - m*x[1]

    x2 = np.array([0])
    y2 = np.array([0])

    m = 14.
    c = -17.


    #grid = [-5, 5, -5, 5]
    grid = [-10, 30, -200, 1400]
    plt.axis(grid)

    plt.plot(x, y, 'o', label='Original data', markersize=10)

    plt.plot(x1, y1, 'ko')
    plt.plot(x2, y2, 'ko')

    x[0] = -5
    x[1] = 25
    plt.plot(x, m*x + c, 'r', label='Fitted line')

    x[0] = -5
    x[1] = 25
    plt.plot(x, (x1[0] - x)/m + y1[0], 'r', label='Fitted line')

    plt.legend()
    plt.show()


def main():

#    fit_line()
#    return

    # generate perfect input data
    n_samples = 500
    n_inputs = 1
    n_outputs = 1

    x = 20*np.random.random((n_samples,n_inputs) )
    perfect_fit = 60*np.random.normal(size=(n_inputs,n_outputs) )

    print 'perfect_fit = ', perfect_fit

    y = scipy.dot(x,perfect_fit)
    assert y.shape == (n_samples,n_outputs)

    # add a little gaussian noise (linear least squares alone should handle this well)
    x_noise = x + np.random.normal(size=x.shape)
    y_noise = y + np.random.normal(size=y.shape)

    # add some outliers to the point set
    n_outliers = 2  #100
    indices = np.arange( x_noise.shape[0] )
    np.random.shuffle(indices)
    outlier_indices = indices[:n_outliers]

    x_noise[outlier_indices] = 20*np.random.random((n_outliers,n_inputs) )
    y_noise[outlier_indices] = 50*np.random.normal(size=(n_outliers,n_outputs) )

    # setup model
    all_data = np.hstack( (x_noise,y_noise) )
    input_columns = range(n_inputs)  # the first columns of the array
    output_columns = [n_inputs + i for i in range(n_outputs)]  # the last columns of the array

    # set up the linear least squares object
    debug = False
    model = LinearLeastSquaresModel(input_columns, output_columns, debug=debug)

    # linear least squares fitting
    linear_fit, residuals, rank, s = scipy.linalg.lstsq(all_data[:,input_columns], all_data[:,output_columns])
    print 'linear_fit = ', linear_fit

    # run RANSAC algorithm
    #ransac_fit, ransac_data_indices = ransac(all_data, model, 2, 1000, 7e3, 300, debug=debug, return_all=True)
    ransac_fit, ransac_data_indices = ransac_modified(all_data, model, 2, 1, 7e3, 300)
    print 'ransac_fit = ', ransac_fit

    # plot input points
    plt.plot(x_noise[:,0], y_noise[:,0], marker='o', label='data', color='#00cc00', linestyle='None', alpha=0.3)

    # plot points classified by RANSAC as inliers
    plt.plot(x_noise[ransac_data_indices,0], y_noise[ransac_data_indices,0], marker='o', label='RANSAC data', color='#ff8000', linestyle='None', alpha=0.6)

    # non_outlier_indices = indices[n_outliers:]
    #plt.plot(x_noise[non_outlier_indices,0], y_noise[non_outlier_indices,0], 'k.', label='noisy data')   # noisy input points
    #plt.plot(x_noise[outlier_indices,0], y_noise[outlier_indices,0], 'r.', label='outlier data')         # outliers added to input points

    plt.plot(x[:,0], np.dot(x,ransac_fit)[:,0], label='RANSAC fit', color='#cc0000')     # RANSAC model
    plt.plot(x[:,0], np.dot(x,perfect_fit)[:,0], label='exact system', color='#006600')  # exact value
    plt.plot(x[:,0], np.dot(x,linear_fit)[:,0], label='linear fit', color='#ff8000')     # linear least squares model

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
