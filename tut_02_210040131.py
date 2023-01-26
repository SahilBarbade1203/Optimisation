
import numpy as np
list = []
print("Input tye : Please fed in appropriate 2 dimension variable")
list.append(float(input("Please fed in 1st dimension of this variable :")))
list.append(float(input("Please fed in 2nd dimension of this variable :")))

def func(x_input):
    d = np.array(x_input)
    return 2*(np.exp(d[0]))*d[1] + 3*d[0]*(d[1]**2)

def compute_gradient( func , x_input ):

    f = len(x_input)
    l = np.array(x_input)
    g = np.ones(f , dtype=float)
    for i in range(f):
        h = np.zeros(f)
        h[i] = 1
        g[i] = (func(l + 0.001*h) - func(l - 0.001*h))/(2*0.001)
    return g


delF = compute_gradient(func ,list)
print(f"Gradient : {delF}")

def compute_hessian( func , x_input):
    f = len(x_input)
    l = np.array(x_input)
    okk = np.ones((f,f) , dtype=float)
    for i in range(f):
        for j in range(f):
            h = np.zeros(f)
            h[i] = 1
            k = np.zeros(f)
            k[i] = 1
            k[j] = 1
            x = np.zeros(f)
            x[i] = 1
            x[j] = -1
            if i == j:
                okk[i,j] = (func(l+0.001*h) - 2*func(l) + func(l - 0.001*h) )/(0.001**2)
            else:
                okk[i,j] = (func(l +0.001*k) + func(l - 0.001*k) - func(l - 0.001*x ) - func(l+ 0.001*x))/(4*0.001*0.001)

    return okk
                


del2F = compute_hessian(func ,list)
print(f"Hessian : {del2F}")








