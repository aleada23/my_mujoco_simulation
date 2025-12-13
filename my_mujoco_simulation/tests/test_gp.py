from my_mujoco_simulation.genetic_programming import panda_simb_jac, panda_task, population

population_ = population.Population(10)
population_.init_pop()
curr_pop = population_.return_pop()
print(curr_pop)

for elem in curr_pop:
    population_.add_in_cache(elem)
population_.new_population()
curr_pop = population_.return_pop()
print(curr_pop)
