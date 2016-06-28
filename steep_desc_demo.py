from math import pow
import numpy as np
import time

start=time.clock()

def loadfile(filename):
    file = open(filename)
    data = []
    for lines in file:
        temp = lines.split()
        data.append([float(i) for i in temp])
    return data

#load file with data points (x,y)
filename = 'file_name_here.txt'
data = loadfile(filename)
x = []
y = []
for i in data:
    x = np.append(x,i[0])
    y = np.append(y,i[1])
#average y value and other useful values
len_x = len(x)
len_y = len(y)
x_range = range(0, len_x, 1)
y_range = range(0, len_y, 1)
sum_y = 0
num_y = 0
for i in y_range:
    sum_y = sum_y + y[i]
    num_y = num_y + 1
avg_y = sum_y/num_y
#desired rho and c values in range (0,1)
rho = 0.9
c=0.01
#epsilon - convergence of chi squared and number of iterations
epsilon = 0.01
num_iterations_global = 1000
num_iterations = 10000
#initial step (dimension = number of fit parameters)
delta_init = [10,10,10,10,10,10]
#document entries in log file
file_log=open('log_demo.txt', 'w')

#fit function and its derivatives go here
def func(x,a0,a1,a2,a3,a4,a5):
    return(a0+a1*x + a2*pow(x,2) + a3*pow(x,3) + a4*pow(x,4)+a5*pow(x,5))
def deriv_0(x,a0,a1,a2,a3,a4,a5):
    return 0
def deriv_1(x,a0,a1,a2,a3,a4,a5):
    return x
def deriv_2(x,a0,a1,a2,a3,a4,a5):
    return pow(x,2)
def deriv_3(x,a0,a1,a2,a3,a4,a5):
    return pow(x,3)
def deriv_4(x,a0,a1,a2,a3,a4,a5):
    return pow(x,4)
def deriv_5(x,a0,a1,a2,a3,a4,a5):
    return pow(x,5)
#function to choose delta
def backtracking(params, delta):
    
    num_params = len(params) 
    
    grad = np.array([0,0,0,0,0,0],dtype = 'float_')
    grad_chi = np.array([0,0,0,0,0,0],dtype = 'float_')
    slope = np.array([0,0,0,0,0,0],dtype = 'float_')
    chi_sqrd = 0
        
    for j in x_range:
        X = x[j]
        y_trial = func(X,params[0],params[1],params[2],params[3],params[4],params[5])
        
        diff = (y_trial-y[j])
        avg_diff = y[j]-avg_y
        chi_dummy = pow(diff,2)/pow(avg_diff,2)
        chi_sqrd = chi_sqrd + chi_dummy
        #calculate gradients
        
        grad[0] = deriv_0(X,params[0],params[1],params[2],params[3],params[4],params[5])
        grad[1] = deriv_1(X,params[0],params[1],params[2],params[3],params[4],params[5])
        grad[2] = deriv_2(X,params[0],params[1],params[2],params[3],params[4],params[5])
        grad[3] = deriv_3(X,params[0],params[1],params[2],params[3],params[4],params[5])
        grad[4] = deriv_4(X,params[0],params[1],params[2],params[3],params[4],params[5])
        grad[5] = deriv_5(X,params[0],params[1],params[2],params[3],params[4],params[5])
        chi_deriv = 2*diff/pow(avg_diff,2)
        dummy = grad*chi_deriv
        grad_chi = grad_chi + dummy
        
    for k in range(0,num_params,1):
        slope[k] = -grad_chi[k]
    
    
    for k in range(0,num_params,1):
        error = c*delta[k]*grad_chi[k]*slope[k]
        upper_bound = chi_sqrd + error
        params_new = np.array(params[:],dtype='float_')
        increase = delta[k]*slope[k]
        params_new[k] = params[k] + increase
        lessthan = False
        chi_sqrd_prev = 0
        i = 0
        while lessthan == False:
            chi_sqrd_new = 0
            for j in x_range:
                X = x[j]
                y_new = func(X,params_new[0],params_new[1],params_new[2],params_new[3],params_new[4], params_new[5])
                diff = (y_new-y[j])
                avg_diff = y[j]-avg_y
                chi_new_dummy = pow(diff,2)/pow(avg_diff,2)
                chi_sqrd_new = chi_sqrd_new + chi_new_dummy
            if chi_sqrd_new <= upper_bound:
                lessthan = True
            else:
                delta[k] = rho*delta[k]
                error = c*delta[k]*grad_chi[k]*slope[k]
                upper_bound = chi_sqrd + error
            increase = delta[k]*slope[k]
            params_new[k] = params[k] + increase
            i = i + 1
    print(delta)
    return delta
    
def write_to_file(string):
    file_log.write(string)
    file_log.flush()
    
    
def grad_desc(params):
    chi_sqrd_prev = 0
    converged = False
    num_params = len(params)
    chi_sqrd_prev = 0
    params_prev = np.array(params[:], dtype = 'float_')
    delta_prev = delta_init
    delta = backtracking(params, delta_init)
    for i in range(0,num_iterations,1):
        equal = np.array([False,False,False,False,False,False])
        delta_eq = np.array([False,False,False,False,False,False])
        
        #for i in range(0, num_iterations,1):
        Y_fit = np.array([len_y],dtype='float_')
        chi_sqrd = 0
        grad = np.array([0,0,0,0,0,0],dtype = 'float_')
        grad_chi = np.array([0,0,0,0,0,0],dtype = 'float_')
        slope = np.array([0,0,0,0,0,0],dtype = 'float_')
        for j in x_range:
            increase = [0,0,0,0,0,0]
            X = x[j]
            y_trial = func(X,params[0],params[1],params[2],params[3],params[4], params[5])
            Y_fit = np.append(Y_fit,y_trial)
            diff = y_trial - y[j]
            avg_diff = y[j]-avg_y
            chi_sqrd = chi_sqrd + pow(diff,2)/pow(avg_diff,2)
            #calculate slope of chi-squared function
            chi_deriv = 2*diff/pow(avg_diff,2)

            #calculate gradients
        
            grad[0] = deriv_0(X,params[0],params[1],params[2],params[3],params[4],params[5])
            grad[1] = deriv_1(X,params[0],params[1],params[2],params[3],params[4],params[5])
            grad[2] = deriv_2(X,params[0],params[1],params[2],params[3],params[4],params[5])
            grad[3] = deriv_3(X,params[0],params[1],params[2],params[3],params[4],params[5])
            grad[4] = deriv_4(X,params[0],params[1],params[2],params[3],params[4],params[5])
            grad[5] = deriv_5(X,params[0],params[1],params[2],params[3],params[4],params[5])
            chi_deriv = 2*diff/pow(avg_diff,2)
            dummy = grad*chi_deriv
            grad_chi = grad_chi + dummy
        print('Parameters: ' + str(params) + ' Chi-squared: ' + str(chi_sqrd) + ' Iteration: ' + str(i) + '\n')
        write_to_file('Parameters: ' + str(params) + ' Chi-squared: ' + str(chi_sqrd) + ' Iteration: ' + str(i) + '\n')
        for k in range(0,num_params,1):
            slope[k] = -grad_chi[k]
            increase = delta[k]*slope[k]
            params[k] = params[k] + increase
            if params[k] == params_prev[k]:
                equal[k] = True
            params_prev[k] = params[k]
            if delta[k] == delta_prev[k]:
                delta_eq[k] = True
        if all(equal) and (chi_sqrd == chi_sqrd_prev):
            print('Params are equal!', equal)
            converged = True
            return params, chi_sqrd, converged
        if i == (num_iterations-1):
            
            return params, chi_sqrd, converged
        
        delta_prev = delta
        chi_sqrd_prev = chi_sqrd
        delta = backtracking(params, delta)
        
def main1():
    
    params_init = np.array([1,1,1,1,1,1], dtype = 'float_')

    #open file
    
    f=open('demo.txt','w')
    
    f.write('Initial parameters are: ' + str(params_init) + '\n')
    params, error, didconverge = grad_desc(params_init)
    if didconverge == True:
        print('Chi squared is converged!')
        f.write('Chi squared is converged!\n')
    f.write('Final parameters are: ' + str(params) + '\n' + 'Error is: ' + str(error) + '\n\n')
                        
    f.flush()
    
    end=time.clock()
    num_sec = end-start
    print(num_sec)
    f.write('Running time (seconds): ' + str(num_sec))
    f.close()
    
    
if __name__ == "__main__":
    main1()
file_log.close()