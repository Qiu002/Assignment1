import streamlit as st
import csv
import random
import io

# --- Function to read CSV file and convert to dictionary ---
def read_csv_to_dict(uploaded_file):
    program_ratings = {}
    if uploaded_file is not None:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        reader = csv.reader(stringio)
        header = next(reader)  # Skip header row
        for row in reader:
            if len(row) < 2:
                continue
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
    return program_ratings


# --- Fitness function ---
def fitness_function(schedule, program_ratings):
    total_rating = 0
    for i, program in enumerate(schedule):
        if program in program_ratings:
            total_rating += program_ratings[program][i]
    return total_rating  # no normalization, more visible difference


# --- Selection (tournament selection) ---
def selection(population, fitnesses):
    tournament_size = 3
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]


# --- Crossover ---
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 2)
        child1 = parent1[:point] + [gene for gene in parent2 if gene not in parent1[:point]]
        child2 = parent2[:point] + [gene for gene in parent1 if gene not in parent2[:point]]
        return child1, child2
    else:
        return parent1[:], parent2[:]


# --- Mutation ---
def mutation(schedule, mutation_rate):
    for i in range(len(schedule)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(schedule) - 1)
            schedule[i], schedule[j] = schedule[j], schedule[i]
    return schedule


# --- Genetic Algorithm ---
def genetic_algorithm(program_ratings, population_size, generations, crossover_rate, mutation_rate):
    programs = list(program_ratings.keys())
    population = [random.sample(programs, len(programs)) for _ in range(population_size)]

    for generation in range(generations):
        fitnesses = [fitness_function(individual, program_ratings) for individual in population]
        new_population = []

        # Elitism – keep best individual
        elite_index = fitnesses.index(max(fitnesses))
        new_population.append(population[elite_index][:])

        while len(new_population) < population_size:
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            new_population.append(mutation(child1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutation(child2, mutation_rate))

        population = new_population

    best = max(population, key=lambda ind: fitness_function(ind, program_ratings))
    best_fitness = fitness_function(best, program_ratings)
    return best, best_fitness


# --- Streamlit UI ---
st.title("📺 TV Scheduling using Genetic Algorithm")

uploaded_file = st.file_uploader("Upload the ratings CSV file", type=["csv"])
population_size = st.slider("Population Size", 10, 200, 50)
generations = st.slider("Number of Generations", 10, 500, 100)
crossover_rate = st.slider("Crossover Rate", 0.0, 1.0, 0.8)
mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.05)

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    program_ratings = read_csv_to_dict(uploaded_file)
    best_schedule, best_fitness = genetic_algorithm(program_ratings, population_size, generations, crossover_rate, mutation_rate)

    st.subheader("🧠 Best Schedule Found:")
    for i, program in enumerate(best_schedule):
        st.write(f"Time Slot {i+1}: {program}")

    st.subheader("⭐ Total Rating (Fitness):")
    st.metric(label="Best Fitness", value=round(best_fitness, 2))

    st.caption("Try adjusting mutation and crossover rates to see how the schedule and fitness score change!")
else:
    st.info("Please upload a CSV file to start.")
