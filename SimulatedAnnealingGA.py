from spoc_delivery_scheduling import trappist_schedule
#from SPOC_DS_Evalution import trappist_schedule

import os
import json
import random
import numpy as np

def simulated_annealing_delivery_schedule(initial_solution, max_iterations, initial_temperature, cooling_rate, ts):
    """
    Simulated annealing algorithm for scheduling time windows and asteroid assignments.
    
    Parameters:
    - initial_solution: The initial solution (chromosome).
    - max_iterations: The number of iterations to perform.
    - initial_temperature: The starting temperature for simulated annealing.
    - cooling_rate: The rate at which to cool down the temperature.
    - ts: The trappist_schedule object that contains the data.
    
    Returns:
    - best_solution: The best solution found by simulated annealing.
    """
    
    # Function to compute the acceptance probability for worse solutions
    def acceptance_probability(old_fitness, new_fitness, temperature):
        if new_fitness > old_fitness:
            return 1.0
        else:
            return np.exp((new_fitness - old_fitness) / temperature)


    
     # Function to generate a neighboring solution by making small changes to the current solution
    def get_neighbor(solution,n_stations):
        neighbor = solution.copy()
        idx = random.randint(0, len(neighbor) - 1)

        # If the index is in the range of activity windows
        if idx < n_stations * 2:
            # Modify the time window by assigning a new random time within the allowed range
            neighbor[idx] = random.randint(0, 80)  # Time window between 0 and 80 days
            #repair_activity_windows(neighbor, n_stations)
        else:
            # Modify asteroid assignments
            mod_idx = idx % 3
            if mod_idx == 0:
                # Mutate asteroid_id
                neighbor[idx] = random.randint(1, 340)  # Random asteroid ID
            elif mod_idx == 1:
                # Mutate station_id
                neighbor[idx] = random.randint(1, n_stations)  # Random station ID
            elif mod_idx == 2:
                # Mutate opportunity_id
                neighbor[idx] = random.randint(1, 8)  # Random opportunity ID

        return neighbor

    # Initialize the current solution and its fitness
    current_solution = initial_solution
    current_fitness = custom_fitness(current_solution, ts.flat_db)

    # Keep track of the best solution found so far
    best_solution = current_solution
    best_fitness = current_fitness

    # Set the initial temperature
    temperature = initial_temperature

    # Simulated annealing loop
    for iteration in range(max_iterations):
        # Generate a neighboring solution
        neighbor_solution = get_neighbor(current_solution, n_stations=12)
        neighbor_fitness = custom_fitness(current_solution, ts.flat_db)

        # Determine whether to accept the neighboring solution
        if acceptance_probability(current_fitness, neighbor_fitness, temperature) > random.random():
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness

        # Update the best solution if the current solution is better
        if current_fitness > best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness

        # Reduce the temperature (cooling process)
        temperature *= cooling_rate

        # Logging for each iteration
        print(f"Iteration {iteration}: Best Fitness = {best_fitness}, Current Temperature = {temperature}")

    return best_solution, best_fitness

def roulette_wheel_selection(population, fitness_values):
    # Ensure all fitness values are positive by taking absolute values
    fitness_values = np.abs(fitness_values)

    # Calculate the total fitness (sum of all fitness values)
    total_fitness = np.sum(fitness_values)

    # If total fitness is zero (all fitnesses are zero), assign equal probabilities
    if total_fitness == 0:
        selection_probs = np.ones(len(fitness_values)) / len(fitness_values)
    else:
        # Normalize fitness values to get selection probabilities
        selection_probs = fitness_values / total_fitness

    # Choose an individual based on the calculated probabilities
    selected_index = np.random.choice(len(population), p=selection_probs)
    
    # Return the selected individual
    return population[selected_index]


def boundary_mutation(chromosome, mutation_rate, n_asteroids, n_stations, start_time, end_time):
    if random.random() < mutation_rate:
        idx = random.randint(0, len(chromosome) - 1)
        if idx < n_stations * 2:
            # Mutate with boundary values
            if random.random() < 0.5:
                chromosome[idx] = start_time
            else:
                chromosome[idx] = end_time
            repair_activity_windows(chromosome, n_stations)
        else:
            mod_idx = idx % 3
            # Mutate asteroid assignments
            if mod_idx == 0:
                # Mutate asteroid_id
                chromosome[idx] = random.randint(1, n_asteroids)  # Random asteroid ID

            elif mod_idx == 1:
                # Mutate station_id
                chromosome[idx] = random.randint(1, n_stations)  # Random station ID

            elif mod_idx == 2:
                # Mutate opportunity_id
                chromosome[idx] = random.randint(1, 8) # Random Opportunity ID
    return chromosome

def custom_fitness(chromosome, data):
        # Unpack the chromosome to get activity windows and asteroid assignments
        activity_windows = chromosome[:24]
        assignments = chromosome[24:]

        # Initialize variables for tracking masses
        station_masses = np.zeros((12, 3))  # 12 stations, 3 materials (A, B, C)
        penalty = 0
        max_days = 80  # The 80-day limit for station activity windows

        # Decode the assignments and calculate masses
        for idx in range(0, len(assignments), 3):
            start = idx
            end = idx + 3

            if len(assignments[start:end]) == 3:
                asteroid_id, station_id, opportunity_id = assignments[start:end]
                #print("( ",asteroid_id, ",", station_id , ", ", opportunity_id,")")
            else:
                print(f"Skipping assignment at index {idx} due to insufficient values.")
                continue

            # Ensure station_id is an integer
            station_id = int(station_id)

            t_arrival = mA = mB = mC = 0

            # Extract delivery details from the data
            try:
                if (asteroid_id, station_id, opportunity_id) in data:
                    opportunity = data[(asteroid_id, station_id, opportunity_id)]
                    t_arrival = opportunity[0]
                    mA = opportunity[1]
                    mB = opportunity[2]
                    mC = opportunity[3]
                else:
                    pass
            except (KeyError, IndexError):
                print("Invalid triplets", (asteroid_id, station_id, opportunity_id))
                # Penalty for invalid asteroid, station, or opportunity
                

            # Check if the delivery is within the station's activity window
        
            if 0 <= 2 * (station_id - 1) < len(activity_windows):
                T_ki = activity_windows[2 * (station_id - 1)]
                T_kf = activity_windows[2 * (station_id - 1) + 1]
            else:
                #print(f"Invalid station_id or activity_windows index: {station_id}")
                continue

            if not (T_ki <= t_arrival <= T_kf):
                # Penalty for delivery outside the activity window
                penalty += 0.0005

            # Now, check the gap between consecutive stations
            for j in range(1, 11):
                #T_jf = activity_windows[2 * (j - 1) + 1]  # Final time of station j
                T_jf = activity_windows[2 * j]  # Final time of station j
                T_ki = activity_windows[2 * j + 1]  # Initial time of station k (next station)
                

                if T_ki - T_jf <= 1:
                    penalty += 0.5  # Penalty for time gap violation

            # Check if the station activity window is within 80 days and valid
            if not (0 <= T_ki <= T_kf <= max_days):
                penalty += 0.01  # Penalty for station window outside allowed range

            # Add the masses to the respective station
            station_masses[station_id - 1, 0] += mA
            station_masses[station_id - 1, 1] += mB
            station_masses[station_id - 1, 2] += mC

        # Calculate minimum mass across stations and materials
        min_mass = np.min(station_masses)

        # Return the fitness with penalties applied (maximize min mass, minimize penalty)
        fitness = min_mass - penalty
        print("Minimum Masses---------", min_mass)
        return fitness

def crossover(parent1, parent2, n_stations):
    # Define crossover points based on the given index ranges
    # Crossover points for each segment
    point1 = random.randint(1, 2*n_stations -1)  # For the first half of time windows (1 to 12)
    point2 = 2*n_stations  # For the second half of time windows (13 to 24)
    n_asteroid = 340
    random_asteroid = random.randint(1,n_asteroid-1 )
    random_point_of_triplets = (random_asteroid*3)+24;
    point3 = random.randint(24, random_point_of_triplets)  
    
    # Create children using 3-point crossover
    child1 = (
        parent1[:point1] + parent2[point1:point2] + 
        parent1[point2:point3] + parent2[point3:]
    )
    child2 = (
        parent2[:point1] + parent1[point1:point2] + 
        parent2[point2:point3] + parent1[point3:]
    )
    # Ensure no overlap in activity windows after crossover
    repair_activity_windows(child1, n_stations)
    repair_activity_windows(child2, n_stations)
    
    return child1, child2

def initialize_population(n_stations, n_asteroids, population_size, ts):
    population = []
    for _ in range(population_size):
        chromosome = []
        i = 0; 
        assignments = []
        time_windows = ts.PoplateTimeWindows()
        #print(time_windows); exit()
        chromosome.extend(time_windows)
        for asteroid_id in range(1, n_asteroids + 1):
            station_id = random.randint(1, n_stations)
           
            '''
            opps = ts.db.get(asteroid_id, {}).get(station_id, [])
            
            opp_elngth = len(opps) 
            if(opp_elngth > 0 ):  # Ensure there are opportunities available for this asteroid/station
                opportunity_id = random.randint(0, opp_elngth)
                chromosome.extend([asteroid_id, station_id, opportunity_id])
            else:
                opportunity_id = 0  # Use a default or invalid opportunity when no opportunities are available
            '''
            max_length = 1020
        while len(assignments) < max_length:
            for asteroid_id in range(1, n_asteroids + 1):
                # Check if we have reached the max length
                if len(assignments) >= max_length:
                    break
                
                # Generate a random station ID
                station_id = random.randint(1, n_stations)
                
                # Fetch available opportunities for the asteroid/station pair
                opportunities = ts.db.get(asteroid_id, {}).get(station_id, [])
                opp_length = len(opportunities)

                # Check if there are opportunities available
                if opp_length > 0:
                    opportunity_id = random.randint(1, opp_length)
                else:
                    # Assign a default opportunity if none are available
                    opportunity_id = 0

                # Create the triplet
                triplet = [asteroid_id, station_id, opportunity_id]

                # Check if the triplet is already in the assignments
                if triplet not in assignments:
                    # Add the unique triplet to the assignments
                    assignments.extend(triplet)

                # If the number of assignments reaches max length, stop
                if len(assignments) >= max_length:
                    break

        chromosome.extend(assignments)

        population.append(chromosome)
       
    return population

def repair_activity_windows(chromosome, n_stations):
    activity_windows = [(chromosome[i], chromosome[i + 1]) for i in range(0, n_stations * 2, 2)]
    sorted_windows = sorted(activity_windows, key=lambda x: x[0])
    
    for i in range(1, len(sorted_windows)):
        if sorted_windows[i][0] < sorted_windows[i - 1][1]:
            sorted_windows[i] = (sorted_windows[i - 1][1] + 1, sorted_windows[i][1])
    
    for i, (start, end) in enumerate(sorted_windows):
        chromosome[i * 2] = start
        chromosome[i * 2 + 1] = end

def main_simulated_annealing():
    n_stations = 12
    n_asteroids = 340
    max_days = 80
    min_gap = 1  # Minimum gap between station windows
    population_size = 100
    generations = 100
    mutation_rate = 0.01
    crossover_rate = 0.8
    initial_temperature = 1000
    cooling_rate = 0.95
    max_iterations = 1000
    
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, 'data', 'spoc', 'scheduling', 'candidates.txt')
    #print(f"Looking for file at: {path}")
    
    # Initialize the class
    ts = trappist_schedule(path=path)
    

    if ts.db is None or not ts.db:
        print("Database was not loaded properly.")
        return
    

    population = initialize_population(n_stations, n_asteroids, population_size, ts)
    best_solution = None
    best_fitness = -float('inf')

    for generation in range(generations):
        #fitness_values = [ts.fitness(individual)[0] for individual in population]
        fitness_values = [custom_fitness(individual, ts.flat_db) for individual in population]
        new_population = []

        all_zeros = all(element == 0 for element in fitness_values)
        if all_zeros:
            continue

        for _ in range(population_size // 2):
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, n_stations)
            else:
                child1, child2 = parent1, parent2

            # Apply swap mutation instead of boundary mutation
            child1 = boundary_mutation(child1, mutation_rate, n_stations, n_asteroids, ts.start_time, ts.end_time)
            child2 = boundary_mutation(child2, mutation_rate, n_stations, n_asteroids, ts.start_time, ts.end_time)
            new_population.extend([child1, child2])
            
        
        population = new_population
        current_best_fitness = max(fitness_values)

        if current_best_fitness in fitness_values:
            current_best_solution = population[fitness_values.index(current_best_fitness)]
        else:
            print(f"Error: Best fitness {current_best_fitness} not found in fitness values.")
            current_best_solution = population[0]  # Default fallback

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution

    # Step 1: Initialize the population
    initial_solution = current_best_solution

    

    # Step 2: Perform simulated annealing
    best_solution, best_fitness = simulated_annealing_delivery_schedule(
        initial_solution, max_iterations, initial_temperature, cooling_rate, ts
    )

    print("Best solution found:", best_solution)

    # Visualize the solution
    ts.plot(best_solution, path="./images/Mass_Simulted_annealing.png")
    ts.plot_time(best_solution, path="./images/time_Simulated_anealing.png")
    ts.pretty(best_solution)

if __name__ == "__main__":
    
    main_simulated_annealing()
