import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def problem_2(G,d,A, B):
    lamda = (np.linalg.inv(A@(np.linalg.inv(G)@(A.T))))@(B + A@(np.linalg.inv(G)@d))
    x = (np.linalg.inv(G))@((A.T)@lamda - d)
    return lamda , x


def QP_activeset(G, d, Aeq, beq, Aineq, bineq, xinit, n, neq, nineq):
    x0 = xinit
    if Aeq != np.empty(shape = (neq , n)):
        if Aineq == np.empty(shape=(nineq , n)) :
            A  = Aeq
            B  = beq
            lamda , x = problem_2(G,d,A,B)
            k = 1
            W_set = np.arange(neq) + 1
            print(f"Iteration Number {k}: x_iterate = {x} , Working_Set = {W_set}")
        else:
            N = 100
            A = np.concatenate((Aeq ,Aineq ) , axis = 0)
            b = np.concatenate((beq,bineq) , axis = 0)
            W_set = np.arange(A.shape[0]) + 1
            for i in range(N):
                l = np.zeros(shape=(A.shape[0],1))
                lamda , p = problem_2(G,d,A,l)
                if p == np.zeros(shape=p.shape):
                    if np.all(lamda > 0) :
                        print(f"Iteration {i} : x = {x0} , Working Set = {W_set}")
                        break
                    else:
                        d = lamda<0
                        j = np.min(lamda[d])
                        l = 0
                        for k in range(len(lamda)):
                            if lamda[k] == j:
                                l = k
                                print(f"Iteration {i} : x = {x0} , Working Set = {W_set}")
                                break
                            else:
                                continue
                        A = np.delete(A , l , 0)
                        b =np.delete(b, l ,0)
                        W_set = np.delete(W_set , l ,0)
                        x0 = x0
                        print(f"Iteration {i} : x = {x0} , Working Set = {W_set}")
                else:
                    z = np.setxor1d(W_set , np.arange(neq + nineq) + 1 )
                    S = np.concatenate((Aeq ,Aineq ) , axis = 0)
                    R = np.concatenate((beq,bineq) , axis = 0)
                    for v in z:
                        S = np.delete(S , v , 0)
                        R = np.delete(R, v , 0)
                    alpha = 1
                    index = 0
                    for b in range(len(R)):
                        ui = (R[b] -( S[b].T @ x0))/(S[b].T @ p)
                        if ui<alpha:
                            alpha = ui
                            index = b
                        else:
                            continue
                    x0 = x0 + alpha*(p)
                    if alpha < 1:
                        A = np.append(S[index] , 0)
                        b = np.append(R[index] , 0)
                        W_set = np.append(W_set , index)
                    else:
                        continue

    else:
        N = 100
        A = Aineq
        b = bineq
        W_set = np.arange(A.shape[0]) + 1
        for i in range(N):
            l = np.zeros(shape=(A.shape[0], 1))
            lamda, p = problem_2(G, d, A, l)
            if p == np.zeros(shape=p.shape):
                if np.all(lamda > 0):
                    print(f"Iteration {i} : x = {x0} , Working Set = {W_set}")
                    break
                else:
                    d = lamda < 0
                    j = np.min(lamda[d])
                    l = 0
                    for k in range(len(lamda)):
                        if lamda[k] == j:
                            l = k
                            print(f"Iteration {i} : x = {x0} , Working Set = {W_set}")
                            break
                        else:
                            continue
                    A = np.delete(A, l, 0)
                    b = np.delete(b, l, 0)
                    W_set = np.delete(W_set, l, 0)
                    x0 = x0
                    print(f"Iteration {i} : x = {x0} , Working Set = {W_set}")
            else:
                z = np.setxor1d(W_set, np.arange(nineq) + 1)
                S = Aineq
                R = bineq
                for v in z:
                    S = np.delete(S, v, 0)
                    R = np.delete(R, v, 0)
                alpha = 1
                index = 0
                for b in range(len(R)):
                    ui = (R[b] - (S[b].T @ x0)) / (S[b].T @ p)
                    if ui < alpha:
                        alpha = ui
                        index = b
                    else:
                        continue
                x0 = x0 + alpha * (p)
                if alpha < 1:
                    A = np.append(S[index], 0)
                    b = np.append(R[index], 0)
                    W_set = np.append(W_set, index)
                    print(f"Iteration {i} : x = {x0} , Working Set = {W_set}")
                else:
                    print(f"Iteration {i} : x = {x0} , Working Set = {W_set}")
                    continue

G = np.array([[2,0],
              [0,2]])
d = np.array([[-2,-5]]).T
Aeq = np.empty(shape=(0,2))
beq = np.empty(shape = (0,1))
Aineq = np.array([[1,-2],
                  [-1,-2],
                  [-1,2],
                  [1,0],
                  [0,1]])
bineq = np.array([[-2,-2,-2,0,0]]).T
n = 2
neq = 0
nineq = 5
xinit = np.array([[2,0]]).T
QP_activeset(G, d, Aeq, beq, Aineq, bineq, xinit, n, neq, nineq)



























