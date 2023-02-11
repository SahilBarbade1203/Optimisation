"""=================================================== Assignment 4 ===================================================

Some instructions:
    * You can write seperate function for gradient and hessian computations.
    * You can also write any extra function as per need.
    * Use in-build functions for the computation of inverse, norm, etc.

"""

""" Import the required libraries"""

# Start your code here
import  numpy as np
import matplotlib.pyplot as plt
import math



# End your code here


def func(x_input):
    """
    --------------------------------------------------------
    Write your logic to evaluate the function value.

    Input parameters:
        x: input column vector (a numpy array of n dimension)

    Returns:
        y : Value of the function given in the problem at x.

    --------------------------------------------------------
    """

    # Start your code here
    y = -0.0001*(abs(np.sin(x_input[0]) * np.sin(x_input[1])*np.exp(abs(100 - (np.sqrt(x_input[0]**2 + x_input[1]**2))/math.pi)))**0.1)
    # End your code here

    return y


def gradient(func, x_input):
    """
    --------------------------------------------------------------------------------------------------
    Write your logic for gradient computation in this function. Use the code from assignment 2.

    Input parameters:
      func : function to be evaluated
      x_input: input column vector (numpy array of n dimension)

    Returns:
      delF : gradient as a column vector (numpy array)
    --------------------------------------------------------------------------------------------------
    """
    # Start your code here
    # Use the code from assignment 2
    h = 0.001
    grad_f = np.array([])
    for i in range(len(x_input)):
        e = np.array([np.zeros(len(x_input), dtype=int)]).T
        e[i][0] = 1
        del_f = (func(x_input + (h * e)) - func(x_input - (h * e))) / (2 * h)
        grad_f = np.append(grad_f, del_f)
    delF = np.array([grad_f]).T


    # End your code here

    return delF


def hessian(func, x_input):
    """
    --------------------------------------------------------------------------------------------------
    Write your logic for hessian computation in this function. Use the code from assignment 2.

    Input parameters:
      func : function to be evaluated
      x_input: input column vector (numpy array)

    Returns:
      del2F : hessian as a 2-D numpy array
    --------------------------------------------------------------------------------------------------
    """
    # Start your code here
    # Use the code from assignment 2
    n = len(x_input)
    del_x = np.full(shape=n, fill_value=0.001)
    del2F = np.array([]).reshape(0, n)
    for i in range(n):
        hess_f = np.array([])
        del_i = np.array([np.zeros(n)]).T
        del_i[i][0] = del_x[i]
        for j in range(n):
            del_j = np.array([np.zeros(n)]).T
            del_j[j][0] = del_x[j]
            if (i == j):
                a = x_input + del_i
                b = x_input - del_j
                value = (func(a) - (2 * func(x_input)) + func(b)) / (del_x[i] ** 2)
                hess_f = np.append(hess_f, value)
            else:
                a = x_input + del_i + del_j
                b = x_input - del_i - del_j
                c = x_input - del_i + del_j
                d = x_input + del_i - del_j
                value = (func(a) + func(b) - func(c) - func(d)) / (4 * del_x[i] * del_x[j])
                hess_f = np.append(hess_f, value)
        del2F = np.vstack([del2F, hess_f])


    # End your code here

    return del2F



def backtracking(gradient ,func ,p, x):

     o = 5   #alpha bar value setting
     v = 0.8 #rho value setting
     c = 0.1 #c value setting
     okk = o # setting alpha value
     l = gradient(func , x)
     i = 0
     while (func(x + okk*p) > (func(x) + c*okk*((gradient(func,x).T)@p))):
         i = i+1
         okk = v* okk
     return okk

def steepest_descent(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for steepest descent using in-exact line search.

    Input parameters:
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector(numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """

    # Start your code here
    N = 15000
    e = 10**(-6)
    x = x_initial
    listx1 = []
    listx2 = []
    fun = []
    j = 0
    for i in range(N):
        j += 1
        alpha = backtracking(gradient , func , -1*gradient(func,x) , x)
        listx1.append(x[0])
        listx2.append(x[1])
        fun.append((func(x)))
        x = x - (alpha*(gradient(func ,x)))
        c = gradient(func, x).T @ gradient(func, x)
        if c < e:
            break
        else:
            continue
    x_output = x
    f_output = func(x_output)
    grad_output = gradient(func , x_output)
    iterate = np.arange(i+1) + 1
    # End your code here

    return x_output, f_output, grad_output,listx1, listx2 ,iterate ,fun


def newton_method(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for newton method using in-exact line search.

    Input parameters:
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector (numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """
    # Start your code here
    N = 15000
    e = 10**(-6)
    x = x_initial
    listx1 = []
    listx2 = []
    fun = []
    for i in range(N):
        alpha = backtracking(gradient , func ,  -1*((np.linalg.inv(hessian(func,x)))@gradient(func,x)) , x)
        listx1.append(x[0])
        listx2.append(x[1])
        fun.append((func(x)))
        x = x - alpha*((np.linalg.inv(hessian(func,x)))@gradient(func,x))
        c = gradient(func, x).T @ gradient(func, x)
        if c<e:
            break
        else:
            continue

    x_output = x
    f_output = func(x_output)
    grad_output = gradient(func,x_output)
    iterate = np.arange(i+1) + 1

    # End your code here

    return x_output, f_output, grad_output,listx1, listx2 ,iterate ,fun


def updation(grad1 , grad2 , x2 , x1 ,B):
    a = (grad2 - grad1) - (B@(x2 -x1))
    S = B + ((a@(a.T))/((a.T)@(x2-x1)))
    return  S


def quasi_newton_method(func, x_initial):
    """
    -----------------------------------------------------------------------------------------------------------------------------
    Write your logic for quasi-newton method with in-exact line search.

    Input parameters:
        func : input function to be evaluated
        x_initial: initial value of x, a column vector (numpy array)

    Returns:
        x_output : converged x value, a column vector (numpy array)
        f_output : value of f at x_output
        grad_output : value of gradient at x_output, a column vector (numpy array)
    -----------------------------------------------------------------------------------------------------------------------------
    """

    # Start your code here
    N =15000
    e = 10**(-6)
    x =x_initial
    listx1 = []
    listx2 = []
    fun = []
    B = np.identity(2 ,dtype=float)
    for i in range(N):
        alpha = backtracking(gradient , func , -1*((np.linalg.inv(B))@(gradient(func ,x))) , x)
        grad1 = gradient(func,x)
        listx1.append(x[0])
        listx2.append(x[1])
        fun.append((func(x)))
        x1 =x
        x = x - alpha*((np.linalg.inv(B))@(gradient(func ,x)))
        grad2 = gradient(func ,x)
        x2 =x
        B = updation(grad1 ,grad2 ,x2, x1 , B)
        c = gradient(func, x).T @ gradient(func, x)
        if c < e:
            break
        else:
            continue

    x_output = x
    f_output = func(x_output)
    grad_output = gradient(func, x_output)
    iterate = np.arange(i+1)+1


    # End your code here

    return x_output, f_output, grad_output ,listx1, listx2 ,iterate ,fun


def iterative_methods(func, x_initial):
    """
     A function to call your steepest descent, newton method and quasi-newton method.
    """
    x_SD, f_SD, grad_SD, list1_SD,list2_SD, iterate_SD,fun_SD = steepest_descent(func, x_initial)
    x_NM, f_NM, grad_NM, list1_NM,list2_NM, iterate_NM ,fun_NM= newton_method(func, x_initial)
    x_QN, f_QN, grad_QN, list1_QN,list2_QN, iterate_QN, fun_QN= quasi_newton_method(func, x_initial)

    return x_SD, f_SD, grad_SD,x_NM, f_NM, grad_NM ,x_QN, f_QN, grad_QN


def plot_x_iterations():
    x_SD, f_SD, grad_SD, list1_SD, list2_SD, iterate_SD, fun_SD = steepest_descent(func, x_initial)
    x_NM, f_NM, grad_NM, list1_NM, list2_NM, iterate_NM, fun_NM = newton_method(func, x_initial)
    x_QN, f_QN, grad_QN, list1_QN, list2_QN, iterate_QN, fun_QN = quasi_newton_method(func, x_initial)


    plt.subplot(1,2,1)
    plt.plot(iterate_SD,list1_SD ,"r")
    plt.plot(iterate_NM,list1_NM ,"black")
    plt.plot(iterate_QN,list1_QN ,"--b")
    plt.xlabel("Number of iterations")
    plt.ylabel("X1 values")
    plt.title("Variation of x1 with iterations")

    plt.subplot(1,2,2)
    plt.plot (iterate_SD,list2_SD, "r")
    plt.plot( iterate_NM,list2_NM, "black")
    plt.plot(iterate_QN,list2_QN, "--b")
    plt.xlabel("Number of iterations")
    plt.ylabel("X2 values")
    plt.title("Variation of x2 with iterations")

    plt.tight_layout()
    plt.show()

def plot_f_iterations():
    x_SD, f_SD, grad_SD, list1_SD, list2_SD, iterate_SD, fun_SD = steepest_descent(func, x_initial)
    x_NM, f_NM, grad_NM, list1_NM, list2_NM, iterate_NM, fun_NM = newton_method(func, x_initial)
    x_QN, f_QN, grad_QN, list1_QN, list2_QN, iterate_QN, fun_QN = quasi_newton_method(func, x_initial)

    plt.plot(iterate_SD, fun_SD, "r")
    plt.plot(iterate_NM, fun_NM, "black")
    plt.plot(iterate_QN, fun_QN, "--b")
    plt.xlabel("Number of iterations")
    plt.ylabel("f(x) values")
    plt.title("Variation of f(x) with iterations")

    plt.tight_layout()
    plt.show()




"""--------------- Main code: Below code is used to test the correctness of your code ---------------

    func : function to evaluate the function value. 
    x_initial: initial value of x, a column vector, numpy array

"""

# Define x_initial here
x_initial = np.array([[1.5, 1.5]]).T

x_SD, f_SD, grad_SD, x_NM, f_NM, grad_NM, x_QN, f_QN, grad_QN = iterative_methods(func, x_initial)

print("\nFor steepest descent:\nFunction converged at x = \n",x_SD)
print("\nFunction value at converged point = \n",f_SD)
print("\nGradient value at converged point = \n",grad_SD)

print("\nFor Newton's Method:\nFunction converged at x = \n",x_SD)
print("\nFunction value at converged point = \n",f_SD)
print("\nGradient value at converged point = \n",grad_SD)

print("\nFor Quasi-Newton's Method:\nFunction converged at x = \n",x_SD)
print("\nFunction value at converged point = \n",f_SD)
print("\nGradient value at converged point = \n",grad_SD)

plot_x_iterations()
plot_f_iterations()