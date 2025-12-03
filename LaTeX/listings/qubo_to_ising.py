import numpy as np

# Funzione per convertire una matrice qubo [creata attraverso numpy] in una matrice di Ising
def qubo_to_ising(qubo):
    if qubo.shape[0] != qubo.shape[1]:
        return None
    
    N = qubo.shape[0] 
    Ising = np.zeros((N, N))

    for i in range(N):
        Ising[i, i] = 0.5 * qubo[i, i]

    for i in range(N):
        for j in range(i + 1, N):
            Ising[i, j] = 0.25 * qubo[i, j]
            Ising[i, i] += 0.25 * qubo[i, j]
            Ising[j, j] += 0.25 * qubo[i, j]

    return Ising