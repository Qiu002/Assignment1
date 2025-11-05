import streamlit as st
import csv
import random

# ------------------------ Streamlit Interface ------------------------
st.title("📊 TV Program Scheduling using Genetic Algorithm")
st.write("This app finds the optimal schedule based on program ratings using a Genetic Algorithm.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your program ratings CSV file", type=["csv"])

# Parameter inputs
st.sidebar.header("Genetic Algorithm Parameters")
CO_R = st.sidebar.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.01)
MUT_R = st.sidebar.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, 0.01)
GEN = st.sidebar.number_input("Generations (GEN)", min_value=50, max_value=1000, value=300, step=50)
POP = st.sidebar.number_input("Population Size (POP)", min_value=10, max_value=500, value=100, step=10)
EL_S = st.sidebar.number_input("Elitism Size (EL_S)", min_value=1, max_value=10, value=3, step=1)

if uploaded_file is not None:
    # ------------------------ Load CSV ------------------------
    def read_csv_to_dict(file):
        program_ratings = {}
        reader = csv.reader(file)
        header = next(reader)
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]
            program_ratings[program] = ratings
        return program_ratings

    ratings = read_csv_to_dict(uploaded_file)

    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))  # time slots from 6 AM to 11 PM

    # ------------------------ Fitness Function ------------------------
    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += ratings[program][time_slot % len(ratings[program])]
        return total_rating

    # ------------------------ Genetic Algorithm ------------------------
    def crossover(schedule1, schedule2):
        crossover_point = random.randint(1, len(schedule1) - 2)
        child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
        child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
        return child1, child2

    def mutate(schedule):
        mutation_point = random.randint(0, len(schedule) - 1)
        new_program = random.choice(all_programs)
        schedule[mutation_point] = new_program
        return schedule

    def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
        population = [initial_schedule.copy() for _ in range(population_size)]
        for i in range(population_size):
            random.shuffle(population[i])

        for generation in range(generations):
            population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
            new_population = population[:elitism_size]

            while len(new_population) < population_size:
                parent1, parent2 = random.choices(population, k=2)
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if random.random() < mutation_rate:
                    child1 = mutate(child1)
                if random.random() < mutation_rate:
                    child2 = mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:population_size]

        return population[0]

    # ------------------------ Run Algorithm ------------------------
    if st.button("Run Genetic Algorithm"):
        st.write("### Running Genetic Algorithm...")
        initial_schedule = all_programs.copy()
        best_schedule = genetic_algorithm(initial_schedule)

        total_rating = fitness_function(best_schedule)

        st.success("✅ Optimal Schedule Found!")
        st.write("### Final Schedule:")
        for time_slot, program in enumerate(best_schedule):
            st.write(f"Time Slot {all_time_slots[time_slot % len(all_time_slots)]:02d}:00 — **{program}**")

        st.write(f"**Total Ratings:** {total_rating:.2f}")

else:
    st.info("Please upload a CSV file to begin.")
