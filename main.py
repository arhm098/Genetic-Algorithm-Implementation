import pickle
import numpy as np
import os
import random
DATASET_DIR = "datasets/"
DATASET_NAME = "dataset1/"

with open(os.path.join(DATASET_DIR, DATASET_NAME, "requirements.pkl"), "rb") as f:
    requirements: np.ndarray = pickle.load(f)

with open(os.path.join(DATASET_DIR, DATASET_NAME, "proficiency_levels.pkl"), "rb") as f:
    proficiency_levels: np.ndarray = pickle.load(f)


# Define function to generate population of solutions 
def generate_population(proficiency_levels, requirements, population_size=100):
    # Get number of projects and students
    num_projects = len(requirements)
    num_students = len(proficiency_levels)
    # Create empty list to hold population
    population = []
    # Loop over population size
    for i in range(population_size):
        # Create empty dictionary to hold solution for each project
        solution = {}
        # Create list to keep track of how many projects each student is assigned to
        students_assigned_counts = [0 for _ in range(num_students)]
        # Create list to keep track of students who are already assigned to 3 projects
        students_full = []
        # Loop over projects
        for project in range(num_projects):
            # Select random number of students (1-3) for project
            selected_students = random.sample(range(num_students-1), random.randint(1, 3))
            # Loop over selected students
            for student in range(len(selected_students)):
                # Check if student is already assigned to 2 projects
                if students_assigned_counts[selected_students[student]] >= 2:
                    # If so, add to list of students who are already assigned to 3 projects
                    students_full += [selected_students[student]]
                    # Loop until a new student is found who is not already assigned to 3 projects or already selected for the current project
                    while True:
                        new_student = random.randint(1, num_students-1)
                        if new_student not in students_full and new_student not in selected_students:
                            selected_students[student] = new_student
                            break
                # Increment count of projects assigned to current student
                students_assigned_counts[selected_students[student]] += 1
            # Add solution for current project to solution dictionary
            solution[project] = selected_students
        # Add solution for current individual to population
        population.append(solution)
    # Return population of solutions
    return population


# Define function to calculate fitness score of a solution
def get_score(solution, proficiency_levels, requirements, fitness_parameter=70):
    # Initialize score to 0
    score = 0
    # Loop over projects in the solution
    for project_name in solution:
        # Get the project requirements and selected students for the current project
        current_project = requirements[project_name]
        selected_students = solution[project_name]
        # Loop over selected students for the current project
        for student in selected_students:
            # Get the proficiency level of the current student
            current_student = proficiency_levels[student]
            # Loop over the skills required for the current project
            for n in range(len(current_project)):
                # Check if the current skill is required for the project and the current student has a proficiency level above the fitness parameter
                if current_project[n] == 1 and current_student[n] > fitness_parameter:
                    # If so, increment the score by 1
                    score += 1
    # Return the total score for the solution
    return score


# Define function to calculate fitness scores for a population of solutions
def get_scores(population, proficiency_levels, requirements, fitness_parameter=70):
    # Initialize list to hold scores
    scores = []
    # Loop over solutions in the population
    for key, current_solution in enumerate(population):
        # Initialize score to 0
        score = 0
        # Loop over projects in the solution
        for project_name in current_solution:
            # Get the project requirements and selected students for the current project
            current_project = requirements[project_name]
            selected_students = current_solution[project_name]
            # Loop over selected students for the current project
            for student in selected_students:
                # Get the proficiency level of the current student
                current_student = proficiency_levels[student]
                # Loop over the skills required for the current project
                for n in range(len(current_project)):
                    # Check if the current skill is required for the project and the current student has a proficiency level above the fitness parameter
                    if current_project[n] == 1 and current_student[n] > fitness_parameter:   
                        # If so, increment the score by 1
                        score += 1
        # Append the score and the solution to the list of scores
        scores.append([score, current_solution])
    # Return the list of scores
    return scores


# Define function to get elite solution from a list of scores and remove it from the list
def get_elites_with_exclusion(scores):
    max = 0
    key = -1
    elite = {}
    # Loop over the solutions in the list of scores
    for _, solution_pair in enumerate(scores):
        # Get the score and the solution
        score = solution_pair[0]
        solution = solution_pair[1]
        # Check if the current score is higher than the previous maximum score
        if score > max:
            # If so, update the maximum score, the key of the solution pair, and the elite solution
            key = _
            max = score
            elite = solution
    # Remove the elite solution from the list of scores and return the elite solution and the updated list of scores
    scores.pop(key)
    return elite, scores

# Define function to get elite solution from a list of scores and remove it from the list
def get_elites_without_exclusion(scores):
    max = 0
    key = -1
    elite = {}
    # Loop over the solutions in the list of scores
    for _,solution_pair in enumerate(scores):
        # Get the score and the solution
        score = solution_pair[0]
        solution = solution_pair[1]
        # Check if the current score is higher than the previous maximum score
        if score > max:
             # If so, update the maximum score, the key of the solution pair, and the elite solution
            key = _
            max = score
            elite = solution
    # scores.pop(key)
    return elite,scores


def crossover(parent1, parent2):
        offspring1 = {}
        offspring2 = {}
        for key in parent1:
            if key % 2 == 0:  # Even keys
                offspring1[key] = parent1[key]  # Inherit from parent1
                offspring2[key] = parent2[key]  # Inherit from parent2
            else:  # Odd keys
                offspring1[key] = parent2[key]  # Inherit from parent2
                offspring2[key] = parent1[key]  # Inherit from parent1
        return offspring1, offspring2


def create_crossover_offspring(input_list):
    # Sort the input_list based on scores
    sorted_list = sorted(input_list, key=lambda x: x[0])
    # Extract the solutions from the sorted list
    solutions = [item[1] for item in sorted_list]
    # Pair up the solutions for crossover
    pairs_list = []
    for i in range(0, len(solutions), 2):
        if i + 1 < len(solutions):
            pairs_list.append((solutions[i], solutions[i + 1]))
        else:
            pairs_list.append((solutions[i], solutions[0]))
    # Create the new generation
    new_gen = []
    for pair in pairs_list:
        child_1, child_2 = crossover(*pair)
        new_gen.append(child_1)
        new_gen.append(child_2)
    # If an extra child was created due to pairing the last and the first solutions, remove it
    if len(new_gen) > len(solutions):
        new_gen.pop()
    return new_gen

# Define function to cause mutations in a population of solutions
def cause_mutation(input_dict, number_of_mutations=43):
    # Loop over the number of mutations to perform
    for i in range(number_of_mutations):
        # Loop over the solutions in the input dictionary
        for key, solution in enumerate(input_dict):
            # Get the size of the current solution
            size_solution = len(solution)
            # Select two random indices in the solution
            random_flip = random.sample(range(size_solution), 2)
            # Swap the values at the two random indices
            solution[random_flip[0]], solution[random_flip[1]] = solution[random_flip[1]], solution[random_flip[0]]
            # Update the solution in the input dictionary
            input_dict[key] = solution
    # Return the input dictionary with mutated solutions
    return input_dict



# Define function to find optimal solutions for a given set of requirements and proficiency levels
def find_optima(proficiency_levels, requirements, population_size=1000, n_iterations=1000, mutation_number=6, fitness_parameter=70, stopping_critera=30):
    
    # Generate an initial population of solutions and calculate the fitness scores for each solution
    current_generation = generate_population(proficiency_levels, requirements, population_size=population_size)
    current_generation = get_scores(current_generation, proficiency_levels, requirements, fitness_parameter=fitness_parameter)
    
    # Get the elite solution from the current generation and its fitness score
    elite, current_generation_with_scores = get_elites_with_exclusion(current_generation)
    elite_score = get_score(elite, proficiency_levels, requirements)
    
    # Print the initial maximum solution
    print("Initial maximum solution:", elite_score)
    
    # Loop over the specified number of iterations
    for i in range(n_iterations):
        
        # Create a new generation of solutions through crossover and mutation
        new_generation = create_crossover_offspring(current_generation_with_scores)
        new_generation = cause_mutation(new_generation, number_of_mutations=mutation_number)
        new_generation = get_scores(new_generation, proficiency_levels, requirements)
        
        # Get the fittest solution from the new generation and its fitness score
        fittest_solution, _ = get_elites_without_exclusion(new_generation)
        fittest_solution_score = get_score(fittest_solution, proficiency_levels, requirements)
        
        # Check if the fittest solution meets the stopping criteria and has a higher fitness score than the current elite solution
        if fittest_solution_score > stopping_critera and elite_score < fittest_solution_score:
            
            # If so, return the fittest solution
            return fittest_solution
        
        # Update the current generation with the new generation
        current_generation = new_generation
    
    # If the function hasn't returned a solution yet, raise an exception
    raise Exception("Solution stuck in local optima. Please change mutation_number or change fitness_parameter or increase population_size.")



def main():
    #intializing stopping critera as 25 and population size and mutation change these if you cant find your solution
    #STOPPING CRITERA SHOULD BE SET DEPENDING ON THE SIZE OF THE DATASET
    #INCREASING MUTATION CAUSES RANDOM CHANGES IN DATA 
    #n_iterations checks for n times if a solution is not found then it stops and raises a exception and asks the user to change the parameters
    sol = find_optima(proficiency_levels,requirements,stopping_critera=30,population_size=1000,mutation_number=2,n_iterations=1000)
    print(sol)

main()