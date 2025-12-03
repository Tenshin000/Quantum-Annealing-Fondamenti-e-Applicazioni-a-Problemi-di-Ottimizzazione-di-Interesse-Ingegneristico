def simulated_annealing(init_temp, final_temp, init_placement, cost_function, perturb_function, schedule_function, inner_loop_criterion):
    """
    Algoritmo Simulated Annealing
    :param init_temp: Temperatura iniziale
    :param final_temp: Temperatura finale
    :param init_placement: Posizionamento (o soluzione iniziale)
    :param cost_function: Funzione per valutare il costo di un posizionamento
    :param perturb_function: Funzione per generare una nuova soluzione
    :param schedule_function: Funzione per aggiornare la temperatura
    :param inner_loop_criterion: Funzione per verificare se il ciclo interno deve terminare
    :return: Il miglior posizionamento trovato
    """
    temp = init_temp
    place = init_placement

    while temp > final_temp:
        while not inner_loop_criterion():
            new_place = perturb_function(place)
            delta_cost = cost_function(new_place) - cost_function(place)

            if delta_cost < 0:
                place = new_place
            else:
                if random.random() > math.exp(-delta_cost / temp):
                    place = new_place
        
        temp = schedule_function(temp)
    
    return place