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

    # Step 1: Initialize the population
    initial_solution = initialize_population(n_stations, n_asteroids, population_size, ts)[0]

    

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
