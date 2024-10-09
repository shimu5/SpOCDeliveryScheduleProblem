
# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
from spoc_delivery_scheduling import trappist_schedule
#from SPOC_DS_Evalution import trappist_schedule

import os
import json
import random
import numpy as np


# Genetic Algorithm Implementation
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

def elitist_selection(population, fitness_values, elite_size):
    """
    Perform elitist selection by selecting the top elite_size individuals based on fitness.
    
    Parameters:
    - population: The current population (list of individuals).
    - fitness_values: The fitness values corresponding to each individual in the population.
    - elite_size: The number of top individuals to select as elites.
    
    Returns:
    - elite_population: The selected elite individuals.
    """
    
    # Sort the population based on fitness values in descending order (higher fitness is better)
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]
    
    # Select the top 'elite_size' individuals
    elite_population = sorted_population[:elite_size]
    
    return elite_population

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

# Corrected custom_fitness function
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

def genetic_algorithm_delivery_schedule():      
    n_stations = 12
    n_asteroids = 340
    max_days = 80
    min_gap = 1
    population_size = 100
    generations = 55
    mutation_rate = 0.01
    crossover_rate = 0.8
    script_dir = os.path.dirname(__file__)
    path = os.path.join(script_dir, 'data', 'spoc', 'scheduling', 'candidates.txt')

    # Initialize the class
    ts = trappist_schedule(path=path)
    
    # Verify that the file was loaded
    if ts.db is None or not ts.db:
        print("Database was not loaded properly.")
        return
    #a = [1,2,3,4,5,6,7,8,9,10,11,12]
    #for idx in range(0, len(a), 3):
    #    start = idx
    #    end = idx + 3
    #    print(start, "-", end, a[start:end])
    #opps = ts.db[2][1]
    #print(len(opps))
    #exit()
    #exit()
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
            parent1,parent2  = elitist_selection(population, fitness_values, 2)
            #parent2 = roulette_wheel_selection(population, fitness_values)
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
        
        print(f"Generation {generation}: Best Fitness = {best_fitness}")


    return best_solution

# Run the genetic algorithm
solution = genetic_algorithm_delivery_schedule()

# Visualization and output
ts = trappist_schedule(path=os.path.join(os.path.dirname(__file__), 'data', 'spoc', 'scheduling', 'candidates.txt'))
print(solution)
ts.plot(solution, path="./images/Elit_Boundary_Mass_.png")
ts.plot_time(x=solution, path="./images/Elit_Boundary_Time_window.png")
ts.pretty(solution)