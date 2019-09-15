import numpy as np
import sys
import copy
import time
import math  # for math.ceil

def sgd(w, error_f, error_gradient_f, fargs=[], n_iterations=100, eval_f=lambda x: x,
        learning_rate=0.001, momentum_rate=0.0, save_wtrace=False, verbose=False):

    w = w.copy()
    
    startTime = time.time()
    startTimeLastVerbose = startTime
    
    nW = len(w)

    wtrace = [w.copy()] if save_wtrace else None
    ftrace = [eval_f(error_f(w, *fargs))]
    
    w_change = 0
    for iteration in range(n_iterations):
        fnow = error_f(w, *fargs)  # to calculate layer outputs for gradient calculation
        grad = error_gradient_f(w, *fargs)
        w_change = -learning_rate * grad + momentum_rate * w_change
        w += w_change
        if save_wtrace:
            wtrace.append(w.copy())
        ftrace.append(eval_f(fnow))

        iterations_per_print = math.ceil(n_iterations/10)
        if verbose and (iteration + 1) % max(1, iterations_per_print) == 0:
            seconds = time.time() - startTimeLastVerbose
            print(f'SGD: Iteration {iteration+1:d} ObjectiveF={eval_f(fnow):.5f} Seconds={seconds:.3f}')
            startTimeLastVerbose = time.time()

    return {'w': w,
            'f': error_f(w, *fargs),
            'n_iterations': iteration,
            'wtrace': np.array(wtrace)[:iteration + 1,:] if save_wtrace else None,
            'ftrace': np.array(ftrace)[:iteration + 1],
            'reason': 'iterations',
            'time': time.time() - startTime}


def adam(w, error_f, error_gradient_f, fargs=[], n_iterations=100, eval_f=lambda x: x,
         learning_rate=0.001, save_wtrace=False, verbose=False):

    w = w.copy()
    
    startTime = time.time()
    startTimeLastVerbose = startTime
    
    beta1 = 0.9
    beta2 = 0.999
    alpha = learning_rate
    epsilon = 10e-8
    nW = len(w)
    g = np.zeros((nW, 1))
    g2 = np.zeros((nW, 1))
    beta1t = beta1
    beta2t = beta2

    wtrace = [w.copy()] if save_wtrace else None
    ftrace = [eval_f(error_f(w, *fargs))]
    
    for iteration in range(n_iterations):
        fnow = error_f(w, *fargs)  # to calculate layer outputs for gradient calculation
        grad = error_gradient_f(w, *fargs)
        g = beta1 * g + (1 - beta1) * grad
        g2 = beta2 * g2 + (1 - beta2) * grad * grad

        beta1t *= beta1
        beta2t *= beta2
        alphat = alpha * np.sqrt(1 - beta2t) / (1 - beta1t)
        # print(w.shape, g.shape, g2.shape)
        w -= alphat * g / (np.sqrt(g2) + epsilon)
        if save_wtrace:
            wtrace.append(w.copy())
        ftrace.append(eval_f(fnow))

        iterations_per_print = math.ceil(n_iterations/10)
        if verbose and (iteration + 1) % max(1, iterations_per_print) == 0:
            seconds = time.time() - startTimeLastVerbose
            print(f'Adam: Iteration {iteration+1:d} ObjectiveF={eval_f(fnow):.5f} Seconds={seconds:.3f}')
            startTimeLastVerbose = time.time()

    return {'w': w,
            'f': error_f(w, *fargs),
            'n_iterations': iteration,
            'wtrace': np.array(wtrace)[:iteration + 1,:] if save_wtrace else None,
            'ftrace': np.array(ftrace)[:iteration + 1],
            'reason': 'iterations',
            'time': time.time() - startTime}

######################################################################
# Scaled Conjugate Gradient algorithm from
#  "A Scaled Conjugate Gradient Algorithm for Fast Supervised Learning"
#  by Martin F. Moller
#  Neural Networks, vol. 6, pp. 525-533, 1993
#
#  Adapted by Chuck Anderson from the Matlab implementation by Nabney
#   as part of the netlab library.
#


def scg(w, error_f, error_gradient_f, fargs=[], n_iterations=100, eval_f=lambda x: x,
        save_wtrace=False, verbose=False):

    float_precision = sys.float_info.epsilon

    w = w.copy()
    sigma0 = 1.0e-6
    fold = error_f(w, *fargs)
    fnow = fold
    gradnew = error_gradient_f(w, *fargs)
    gradold = copy.deepcopy(gradnew)
    d = -gradnew      # Initial search direction.
    success = True    # Force calculation of directional derivs.
    nsuccess = 0      # nsuccess counts number of successes.
    beta = 1.0e-6     # Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15 # Lower bound on scale.
    betamax = 1.0e20  # Upper bound on scale.
    nvars = len(w)
    iteration = 1     # count of number of iterations

    wtrace = [w.copy()] if save_wtrace else None
    ftrace = [eval_f(fold)]

    thisIteration = 1
    startTime = time.time()
    startTimeLastVerbose = startTime

    # Main optimization loop.
    while thisIteration <= n_iterations:

        # Calculate first and second directional derivatives.
        if success:
            mu = d.T @ gradnew
            if mu >= 0:
                d = -gradnew
                mu = d.T @ gradnew
            kappa = d.T @ d
            if math.isnan(kappa):
                print('kappa', kappa)

            if kappa < float_precision:
                return {'w': w,
                        'f': fnow,
                        'n_iterations': iteration,
                        'wtrace': np.array(wtrace)[:iteration + 1, :] if save_wtrace else None, 
                        'ftrace': np.array(ftrace)[:iteration + 1],
                        'reason': 'limit on machine precision',
                        'time': time.time() - startTime}
            sigma = sigma0 / math.sqrt(kappa)

            w_smallstep = w + sigma * d
            error_f(w_smallstep, *fargs)  # forward pass through model for intermediate variable values for gradient
            g_smallstep = error_gradient_f(w_smallstep, *fargs)
            theta = d.T @ (g_smallstep - gradnew) / sigma
            if math.isnan(theta):
                print(f'theta {theta} sigma {sigma} d[0] {d[0]} g_smallstep[0] {g_smallstep[0]} gradnew[0] {gradnew[0]}')

        ## Increase effective curvature and evaluate step size alpha.
        delta = theta + beta * kappa
        if math.isnan(delta):
            print(f'delta is NaN theta {theta} beta {beta} kappa {kappa}')
        elif delta <= 0:
            delta = beta * kappa
            beta = beta - theta / kappa

        if delta == 0:
            success = False
            fnow = fold
        else:
            alpha = -mu / delta
            ## Calculate the comparison ratio Delta
            wnew = w + alpha * d
            fnew = error_f(wnew, *fargs)
            Delta = 2 * (fnew - fold) / (alpha * mu)
            if not math.isnan(Delta) and Delta  >= 0:
                success = True
                nsuccess += 1
                w[:] = wnew
                fnow = fnew
            else:
                success = False
                fnow = fold

        iterations_per_print = math.ceil(n_iterations/10)
        if verbose and thisIteration % max(1, iterations_per_print) == 0:
            seconds = time.time() - startTimeLastVerbose
            print(f'SCG: Iteration {iteration:d} ObjectiveF={eval_f(fnow):.5f} Scale={beta:.3e} Seconds={seconds:.3f}')
            startTimeLastVerbose = time.time()
        if save_wtrace:
            wtrace.append(w.copy())
        ftrace.append(eval_f(fnow))

        if success:

            fold = fnew
            gradold[:] = gradnew
            gradnew[:] = error_gradient_f(w, *fargs)

            # If the gradient is zero then we are done.
            gg = gradnew.T @ gradnew
            if gg == 0:
                return {'w': w,
                        'f': fnow,
                        'n_iterations': iteration,
                        'wtrace': np.array(wtrace)[:iteration + 1, :] if save_wtrace else None,
                        'ftrace': np.array(ftrace)[:iteration + 1],
                        'reason': 'zero gradient',
                        'time': time.time() - startTime}

        if math.isnan(Delta) or Delta < 0.25:
            beta = min(4.0 * beta, betamax)
        elif Delta > 0.75:
            beta = max(0.5 * beta, betamin)

        # Update search direction using Polak-Ribiere formula, or re-start
        # in direction of negative gradient after nparams steps.
        if nsuccess == nvars:
            d[:] = -gradnew
            nsuccess = 0
        elif success:
            gamma = (gradold - gradnew).T @ (gradnew / mu)
            d[:] = gamma * d - gradnew

        thisIteration += 1
        iteration += 1

        # If we get here, then we haven't terminated in the given number of iterations.

    return {'w': w,
            'f': fnow,
            'n_iterations': iteration,
            'wtrace': np.array(wtrace)[:iteration + 1,:] if save_wtrace else None,
            'ftrace': np.array(ftrace)[:iteration + 1],
            'reason': 'did not converge',
            'time': time.time() - startTime}




