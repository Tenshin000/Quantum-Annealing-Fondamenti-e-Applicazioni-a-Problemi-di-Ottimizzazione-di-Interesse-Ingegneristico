import itertools
import networkx as nx
import dimod as di
import pandas as pd
from collections import defaultdict
from dwave.system import LeapHybridSampler

# Il codice prende ispirazione da quello contenuto in:
# https://github.com/dwavesystems/dwave-networkx/blob/main/dwave_networkx/algorithms/tsp.py
# Questo per avere una funzione "tsp_to_qubo" in grado di fornire una matrice Q ottimizzata per l'architettura dei Quantum Annealer della D-Wave
# Questo approccio permette di ridurre il numero di variabili e semplificare la complessita' del problema, rendendolo piu' adatto alla risoluzione 
# tramite Quantum Annealing. 

def tsp_to_qubo(graph, lagrange=None, weight='weight', missing_weight=None):
    """
    Genera il modello QUBO per risolvere il problema del commesso viaggiatore (TSP) su un grafo dato.

    Parametri:
    ----------
    graph : networkx.Graph
        Grafo completo in cui ogni arco ha un attributo di peso associato.

    lagrange : float, opzionale (default None)
        Fattore di penalizzazione per garantire che i vincoli siano rispettati.

    weight : str, opzionale (default 'weight')
        Nome dell'attributo che rappresenta il peso degli archi del grafo.

    missing_weight : float, opzionale (default None)
        Peso da assegnare agli archi mancanti nel caso di un grafo incompleto. Se non specificato, viene usata
        la somma totale dei pesi esistenti nel grafo.

    Restituisce:
    -----------
    qubo : dict
        Dizionario che rappresenta il modello QUBO. Le chiavi sono tuple di variabili, e i valori i coefficienti.
    """
    num_nodes = graph.number_of_nodes()

    if lagrange is None:
        # Determina un valore di default per lagrange basato sulla dimensione e sui pesi del grafo
        if graph.number_of_edges() > 0:
            lagrange = graph.size(weight=weight) * num_nodes / graph.number_of_edges()
        else:
            lagrange = 5

    if num_nodes < 3:
        raise ValueError("Il grafo deve avere almeno 3 nodi.")

    qubo = defaultdict(float)

    # Vincoli di presenza
    for node in graph:
        for pos in range(num_nodes):
            qubo[((node, pos), (node, pos))] -= 2 * lagrange

    # Vincoli di unicita' del nodo
    for node in graph:
        for pos1 in range(num_nodes):
            for pos2 in range(pos1 + 1, num_nodes):
                qubo[((node, pos1), (node, pos2))] += 2 * lagrange

    # Vincoli di unicita' della posizione
    for pos in range(num_nodes):
        for node1 in graph:
            for node2 in set(graph) - {node1}:
                qubo[((node1, pos), (node2, pos))] += lagrange

    # Termini obiettivo
    for u, v in graph.edges():
        cost = graph[u][v][weight]
        for pos in range(num_nodes):
            next_pos = (pos + 1) % num_nodes
            qubo[((u, pos), (v, next_pos))] += cost
            qubo[((v, pos), (u, next_pos))] += cost

    return qubo


def solve_qubo(Q, **sampler_args):
    """
    Risolve un modello QUBO utilizzando un solver esatto (ExactSolver).

    Parametri:
    ----------
    Q : dict
        Modello QUBO da risolvere.

    sampler_args : kwargs
        Argomenti opzionali per il campionatore.

    Restituisce:
    -----------
    sample : dict
        Soluzione del modello QUBO con le variabili binarie e i loro valori.
    """
    sampler = di.ExactSolver()
    response = sampler.sample_qubo(Q, **sampler_args)

    sample = response.first.sample
    print("(Nodo, Posizione) [non contano il vincolo sulla partenza]:")
    for var, value in sample.items():
        print(f"  {var}: {value}")

    print("\n")
    return sample


def solve_tsp_quantum_annealing(Q, start=None, **sampler_args):
    """
    Risolve il modello QUBO per il problema TSP utilizzando il Quantum Annealer ibrido di D-Wave.

    Parametri:
    ----------
    Q : dict
        Modello QUBO da risolvere.

    start : hashable, opzionale
        Nodo da considerare come punto di partenza del percorso.

    sampler_args : kwargs
        Argomenti opzionali per il campionatore.

    Restituisce:
    -----------
    route : list
        Lista ordinata dei nodi che rappresentano il percorso ottimale.
    """
    sampler = LeapHybridSampler()
    response = sampler.sample_qubo(Q, **sampler_args)
    sample = response.first.sample

    route = [None] * len(G)
    for (city, time), val in sample.items():
        if val:
            route[time] = city

    if start is not None and route[0] != start:
        # Ruota il percorso per iniziare dal nodo specificato
        idx = route.index(start)
        route = route[idx:] + route[:idx]

    print("Soluzione: ")
    print(route)
    print("\n")

    return route


def print_matrix(m):
    """
    Stampa un dizionario come matrice leggibile utilizzando pandas.

    Parametri:
    ----------
    m : dict
        Dizionario che rappresenta una matrice.
    """
    variables = sorted(set(var for pair in m.keys() for var in pair))

    matrix = pd.DataFrame(0.0, index=pd.MultiIndex.from_tuples(variables), columns=pd.MultiIndex.from_tuples(variables))

    for (var1, var2), value in m.items():
        matrix.loc[var1, var2] = value

    print(matrix.to_string(index=False, header=False))
    print("\n")


def create_graph(distance_matrix):
    """
    Genera un grafo completo basato su una matrice delle distanze.

    Parametri:
    ----------
    distance_matrix : numpy.ndarray
        Matrice contenente le distanze tra i nodi.

    Restituisce:
    -----------
    G : networkx.Graph
        Grafo completo con pesi sugli archi.
    """
    G = nx.Graph()
    num_nodes = distance_matrix.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = distance_matrix[i, j]
            G.add_edge(i, j, weight=weight)
    return G


def order_graph(graph):
    """
    Ordina i nodi e gli archi di un grafo in modo deterministico.

    Parametri:
    ----------
    graph : networkx.Graph
        Grafo da ordinare.

    Restituisce:
    -----------
    G : networkx.Graph
        Grafo ordinato.
    """
    G = nx.Graph()
    G.add_nodes_from(sorted(graph.nodes(data=True)))
    G.add_edges_from(graph.edges(data=True))
    return G


def print_graph(G):
    """
    Stampa i nodi e gli archi di un grafo.

    Parametri:
    ----------
    graph : networkx.Graph
        Grafo da stampare.
    """
    print("Nodi:", sorted(G.nodes()))
    print("Archi con pesi:")
    for u, v, data in sorted(G.edges(data=True)):
        print(f"({u}, {v}) - peso: {data['weight']}")
    print("\n")



# Risoluzione Problema del TSP

# Creazione Grafo 
G = nx.Graph()
edges = [("A", "B", 20), ("A", "C", 10), ("A", "D", 18), ("B", "C", 12), ("B", "D", 18), ("C", "D", 6)]
G.add_weighted_edges_from(edges)
print_graph(G)

# Creazione QUBO
Q = tsp_to_qubo(G)
print("Matrice QUBO:")
print_matrix(Q)

# Risoluzione Problema
solve_qubo(Q)
solve_tsp_quantum_annealing(Q, start="A")