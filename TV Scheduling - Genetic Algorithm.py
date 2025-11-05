import streamlit as st
import pandas as pd
import random

# ------------------------ Streamlit Interface ------------------------
st.title("📊 TV Program Scheduling using Genetic Algorithm")
st.write("This app finds the optimal schedule based on program ratings using a Genetic Algorithm.")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload your program ratings CSV file", type=["csv"])

# Sidebar parameters
st.sidebar.header("⚙️ Genetic Algorithm Parameters")
CO_R = st.sidebar.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8, 0.01)
MUT_R = st.sidebar.slider("Mutation Rate (MUT_R)", 0.01, 0.1, 0.02, 0.01)
GEN = st.sidebar.number_input("Generations (GEN)", min_value=50, max_value=1000, value=300, step=50)
POP = st.sidebar.number_input("Population Size (POP)", min_value=10, max_value=500, value=100, step=10)
EL_S = st.sidebar.number_input("Elitism Size (EL_S)", min_value=1, max_value=10, value=3, step=1)


# ==========================================================
# 🧩 MAIN CODE EXECUTION
# ==========================================================
if uploaded_file is not None:
    # ------------------------ Load CSV ------------------------
    def read_csv_to_dict(uploaded_file):
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"⚠️ Error reading CSV file: {e}")
            return {}

        if df.empty:
            st.error("⚠️ The uploaded CSV file is empty.")
            return {}

        if df.shape[1] < 2:
            st.error("⚠️ The CSV must have at least one program column and one rating column.")
            return {}

        program_ratings = {}
        for _, row in df.iterrows():
            program = str(row.iloc[0])
            try:
                ratings = [float(x) for x in row.iloc[1:].tolist()]
            except ValueError:
                st.error(f"⚠️ Invalid numeric value found in program '{program}'.")
                return {}
            program_ratings[program] = ratings

        # Preview
        st.write("### 📄 CSV Preview (first 5 rows):")
        st.dataframe(df.head())

        return program_ratings

    ratings = read_csv_to_dict(uploaded_file)

    if ratings:  # Only proceed if CSV is valid
        all_programs = list(ratings.keys())
        all_time_slots = list(range(6, 24))  # 6 AM to 11 PM

        # Determine global min/max possible fitness for normalization
        global_min = min([min(v) for v in ratings.values()])
        global_max = max([max(v) for v in ratings.values()])

        # ------------------------ Fitness Function ------------------------
        def fitness_function(schedule):
            total_rating = 0
            for time_slot, program in enumerate(schedule):
                weight = (time_slot + 1) / len(schedule)
                total_rating += ratings[program][time_slot % len(ratings[program])] * weight
            return total_rating

        # ------------------------ GA Operators ------------------------
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

        def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP,
                              crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S):
            population = []
            for _ in range(population_size):
                individual = [random.choice(all_programs) for _ in range(len(all_time_slots))]
                population.append(individual)

            best_overall = None
            best_fitness_value = float("-inf")

            for generation in range(generations):
                fitness_scores = [fitness_function(ind) for ind in population]
                sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
                population = [population[i] for i in sorted_indices]
                fitness_scores.sort(reverse=True)

                if fitness_scores[0] > best_fitness_value:
                    best_fitness_value = fitness_scores[0]
                    best_overall = population[0]

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

            # Normalize based on global range (not per generation)
            normalized_best = (best_fitness_value - global_min * len(all_time_slots)) / (
                (global_max - global_min) * len(all_time_slots)
            )
            normalized_best = max(0, min(1, normalized_best))  # keep between 0–1

            return best_overall, normalized_best

        # ------------------------ Run Algorithm ------------------------
        if st.button("▶️ Run Genetic Algorithm"):
            st.write("### Running Genetic Algorithm... Please wait...")
            initial_schedule = all_programs.copy()
            best_schedule, total_rating = genetic_algorithm(initial_schedule)

            # ------------------------ Display Results ------------------------
            st.success("✅ Optimal Schedule Found!")
            st.write("### 🗓️ Final Optimal Schedule")

            schedule_data = {
                "Time Slot": [f"{all_time_slots[i % len(all_time_slots)]:02d}:00" for i in range(len(best_schedule))],
                "Program": best_schedule
            }
            schedule_df = pd.DataFrame(schedule_data)
            st.table(schedule_df)
            st.write(f"### ⭐ Normalized Total Rating: {total_rating:.4f} (0–1 scale)")

else:
    st.info("📥 Please upload a CSV file to begin.")
