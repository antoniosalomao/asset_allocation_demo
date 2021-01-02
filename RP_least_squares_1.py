import pandas as pd
import numpy as np
import math
import scipy.optimize
from scipy.optimize import minimize
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
from matplotlib.pyplot import figure
import seaborn as sns

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#----------------------#
# Matplotlib Functions #
#----------------------#

def get_scatter_ticker_plot(x_data, y_data, tick_data, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x_data, y_data, marker='x', color='r')
    plt.title(title, size=12.5)
    ax.set_xlabel(x_label), ax.set_ylabel(y_label)
    for i, txt in enumerate(tick_data):
        ax.annotate(' {:.4f}'.format(txt), (x_data[i], y_data[i]))
    plt.tight_layout()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------#
# Risk Parity - Helper functions #
#--------------------------------#

def RP_weight_constraint(x0, Q, C):
    '''
    Sum(weights) constraint (C)
    '''
    sum_X = sum(x0[:len(Q)]) - C

    return sum_X

def RP_objective_function_simple(x0, Q):
    '''
    Equal Risk Contribution Risk Parity (Positive weights only), method: Least-Squares
    '''
    X, theta = x0[:Q.shape[0]], x0[-1]
    F = sum([math.pow(X[N]*Q[N]@X - theta, 2)for N, i in enumerate(X)])

    return F

def RP_objective_function_minvar_ls(x0, Q, RHO):
    '''
    Minimum Variance Risk Parity, method: Least-Squares
    '''
    X, theta = x0[:Q.shape[0]], x0[-1]
    F = sum([math.pow(X[N]*Q[N]@X - theta, 2) for N, i in enumerate(X)]) + RHO*X.T@Q@X

    return F

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------------------#
# Portfolio Metrics - Helper Functions #
#--------------------------------------#

def get_RP_weights(Q, report):
    '''
    RP weights
    '''
    RP_weights = np.array(report.x[:len(Q)])

    return RP_weights

def get_expected_return(X, R):
    '''
    Expected Returns: E[R]
    '''
    expected_return = X@R

    return expected_return

def get_port_variance(X, Q):
    '''
    Portfolio Variance
    '''
    port_variance = X@Q@X.T

    return port_variance

def get_rc_rrc(X, Q):
    '''
    Risk Contribution
    Relative Risk Contribution
    '''
    rc = np.array([X[N]*Q[N]@X for N, i in enumerate(X)])
    rrc = np.array([k/sum(rc) for k in rc])
    rc_rrc = tuple((rc, rrc))

    return rc_rrc

def get_sharpe(X, R, RF,Q):
    '''
    Sharpe Ratio
    '''
    sharpe = ((X.T@R) - RF)/ (np.power(X.T@Q@X, 0.5))

    return sharpe

def get_herfidhal_index(X, Q):
    '''
    Herfindhal Index
    '''
    hi = sum([math.pow(((X[N]*Q[N]@X)/(X.T@Q@X)), 2) for N, i in enumerate(X)])

    return hi

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------#
# ScipyOptimize - Helper Functions #
#----------------------------------#

def get_best_report_over_n_trials(n_trials, opt_dict):
    '''
    N trials - ScipyOptimize - Minimize
    '''
    reports = []
    for _ in range(n_trials):
        opt_report = minimize(**opt_dict, tol=math.pow(10, -16), options={'maxiter': math.pow(10, 6)})
        if (opt_report.success == True):
            reports.append(tuple((float(opt_report.fun), opt_report)))
    if len(reports) > 0:
        best_opt_report = sorted(reports, key=lambda item: item[0])[0][1]
    else:
        best_opt_report = {}

    return best_opt_report


def get_init_X_SMVRP(report, all_solutions, Q, LB_UB_x, RHO_i):
    '''
    Initial X array - Sequential min-var RP
    '''
    if (bool(report) == False):
        if (len(all_solutions) == 0):
            init_X = np.random.uniform(low=LB_UB_x[0], high=LB_UB_x[1], size=len(Q))
        else:
            init_X = all_solutions[-1][1]
    else:
        all_solutions.append(tuple((RHO_i, report.x[:len(Q)], report)))
        init_X = all_solutions[-1][1]

    return init_X

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------#
# Sequential Minimum Variance Risk Parity Algorithm #
#---------------------------------------------------#

def get_SMVRP_report(Q, LB_UB_x, C, RHO_tol, RHO_n_trials):
    '''
    Sequential Minimum Variance Risk Parity
    '''
    RHO = sorted([math.pow(2, i) for i in np.arange(-20, 18)], reverse=True)
    all_solutions = []
    for N, RHO_i in enumerate(RHO):
        if (N == 0):
            init_X = np.random.uniform(low=LB_UB_x[0], high=LB_UB_x[1], size=len(Q))
        elif (RHO_i < RHO_tol):
            RHO_i = 0   
        rp_opt_dict = {'Q': Q, 'LB_UB_x': LB_UB_x, 'C': C, 'init_X': init_X, 'RHO_i': RHO_i}
        report = get_RP_minvar_ls_report(**rp_opt_dict)
        init_X = get_init_X_SMVRP(report=report, all_solutions=all_solutions, Q=Q, LB_UB_x=LB_UB_x, RHO_i=RHO_i)
        print('Herfindhal index: {hi:.4f} ( Target: {t} )'.format(hi=get_herfidhal_index(X=init_X, Q=Q), t=(1/len(Q))))
    report = sorted(all_solutions, key=lambda x: x[0])[0][-1]

    return report

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#--------------------------#
# Risk Parity Optimization #
#--------------------------#

def get_RP_simple_report(Q, LB_UB_x, C):
    '''
    Scipy Report (Risk Parity)
    '''
    init_guess_X_theta = np.array(list(np.random.uniform(low=LB_UB_x[0], high=LB_UB_x[1], size=len(Q))) + \
                                  list(np.random.uniform(low=-10, high=10, size=1)))
    opt_dict = {'fun': RP_objective_function_simple,
                 'x0': init_guess_X_theta,
               'args': Q,
             'bounds': [LB_UB_x for _ in Q] + [(-np.inf, np.inf)],
        'constraints': {'type': 'eq', 'fun': RP_weight_constraint, 'args': tuple((Q, C))}}
    best_report = get_best_report_over_n_trials(n_trials=20, opt_dict=opt_dict)

    return best_report

def get_RP_minvar_ls_report(Q, LB_UB_x, C, init_X, RHO_i):
    '''
    Scipy Report (Risk Parity) - Minimum Variance
    '''
    opt_dict = {'fun': RP_objective_function_minvar_ls,
                 'x0': np.array(list(init_X) + list(np.random.uniform(low=-10, high=10, size=1))),
               'args': tuple((Q, RHO_i)),
             'bounds': [LB_UB_x for _ in Q] + [(-np.inf, np.inf)],
        'constraints': {'type': 'eq', 'fun': RP_weight_constraint, 'args': tuple((Q, C))}}
    best_report = get_best_report_over_n_trials(n_trials=20, opt_dict=opt_dict)

    return best_report

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#----------------#
# Main Functions #
#----------------#

def get_complete_rp_solutions(port_dict, n_trials, rp_type):
    '''
    Returns a dictionary with all RP solutions
    '''
    R, Q, LB_UB_x, C = port_dict['R'], port_dict['Q'], port_dict['LB_UB_x'], port_dict['C']
    RP_solutions_report = {'Portfolio N': [], 'Report': [], 'X': [], 'E[R]': [], 'Variance': [], 'RC': [], 'RRC': [], 'Herfindhal index':[], 'Sharpe': []}
    RP_solutions_report.update(port_dict)
    for trial in range(n_trials):
        if (rp_type == 'Random'):
            report = get_RP_simple_report(Q=Q, LB_UB_x=LB_UB_x, C=C)
        elif (rp_type == 'Min-variance'):
            report = get_SMVRP_report(Q=Q, LB_UB_x=LB_UB_x, C=C, RHO_tol=0.000005, RHO_n_trials=10)
        elif (rp_type == 'Max-Sharpe'):
            print('Max Sharpe')
        else:
            print('')
        if (bool(report) == True):
            X = get_RP_weights(Q=Q, report=report)
            E_R, port_variance = get_expected_return(X=X, R=R), get_port_variance(X=X, Q=Q)
            rc, rrc = get_rc_rrc(X=X, Q=Q)
            hi, sharpe = get_herfidhal_index(X=X, Q=Q), get_sharpe(X=X, R=R, RF=0, Q=Q)
            RP_solutions_report['Portfolio N'].append(trial + 1),  RP_solutions_report['Report'].append(report)
            RP_solutions_report['X'].append(X),                    RP_solutions_report['E[R]'].append(E_R)
            RP_solutions_report['Variance'].append(port_variance), RP_solutions_report['RC'].append(rc)
            RP_solutions_report['RRC'].append(rrc),                RP_solutions_report['Herfindhal index'].append(hi)
            RP_solutions_report['Sharpe'].append(sharpe)
            print('====== Trial #{} ======'.format(trial + 1))
            print('RP weights: {}'.format(X))
            print('RRC: {}'.format(rrc))
            print('Sharpe: {}'.format(sharpe))
            print('\n')

    return RP_solutions_report

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#------#
# Main #
#------#

covar_k1 = np.array([[94.868, 33.750, 12.325, -1.178, 8.778],
                     [33.750, 445.642, 98.955, -7.901, 84.954],
                     [12.325, 98.955, 117.265, 0.503, 45.184],
                     [-1.178, -7.901, 0.503, 5.460, 1.057],
                     [8.778, 84.954, 45.184, 1.057, 34.126]])

ret_k1 = np.array([0.15, -0.05, 0.1, -0.07, 0.25])

port_dict_1 = {'R': ret_k1, 'Q': covar_k1}
cons_dict = {'LB_UB_x': tuple((-1, 1)), 'C': 1.5}
port_dict_1.update(cons_dict)

all_solutions = get_complete_rp_solutions(port_dict=port_dict_1, n_trials=1, rp_type='Min-variance')
print(all_solutions)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------#
#------------#
# Matplotlib #
#------------#

# Multiple RP solutions
trials = [v for k, v in all_solutions.items() if (k == 'Portfolio N')][0]
expected_returns = [v for k, v in all_solutions.items() if (k == 'E[R]')][0]
sharpe = [v for k, v in all_solutions.items() if (k == 'Sharpe')][0]

'''get_scatter_ticker_plot(x_data=trials, y_data=expected_returns, tick_data=sharpe, 
                        x_label='Trial N', y_label='Expected Return', title ='RP trials')
plt.show()
'''

























