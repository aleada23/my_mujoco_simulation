import numpy as np
import copy
import random

class Population:
    def __init__(self, size):
        self.size = size
        self.tasks = ["IK", "Manip", "MJL"]

        self.current_pop = []
        self.cache = []
        self.mask = None

    # ------------------------------------------------------
    # TASK GENERATOR
    # ------------------------------------------------------
    def make_random_task(self, name=None):
        if name is None:
            name = np.random.choice(self.tasks)

        present = np.random.random() < 0.85

        if name == "IK":
            return [name, present,
                    np.random.random()*2,
                    np.random.random()*10]

        if name == "Manip":
            return [name, present,
                    np.random.random()*100]

        if name == "MJL":
            return [name, present,
                    np.random.random()*100]

        raise ValueError("Unknown task type:", name)

    # ------------------------------------------------------
    # POPULATION INIT
    # ------------------------------------------------------
    def init_pop(self):
        self.current_pop = []

        for _ in range(self.size):
            ind = []

            # random task order
            task_order = self.tasks.copy()
            np.random.shuffle(task_order)

            for t in task_order:
                ind.append(self.make_random_task(t))

            # cost at index 0 (GP-style)
            ind.insert(0, -1)
            self.current_pop.append(ind)

    def return_pop(self):
        return copy.deepcopy(self.current_pop)

    # ------------------------------------------------------
    # CROSSOVER (ORDER FROM P1, RANDOM PARAM INHERITANCE)
    # ------------------------------------------------------
    def crossover(self, h1, h2):
        parent1 = copy.deepcopy(self.current_pop[h1])
        parent2 = copy.deepcopy(self.current_pop[h2])

        # skip cost field
        tasks1 = parent1[1:]
        tasks2 = parent2[1:]

        # map name->task for parent2
        p2map = {t[0]: t for t in tasks2}

        child = []

        c_from_p1 = False
        c_from_p2 = False

        for t1 in tasks1:
            name = t1[0]

            if np.random.random() < 0.5:
                child.append(copy.deepcopy(t1))
                c_from_p1 = True
            else:
                child.append(copy.deepcopy(p2map[name]))
                c_from_p2 = True

        # enforce contribution from both parents
        if not c_from_p1:
            idx = np.random.randint(0, len(child))
            child[idx] = copy.deepcopy(tasks1[idx])

        if not c_from_p2:
            idx = np.random.randint(0, len(child))
            name = tasks1[idx][0]
            child[idx] = copy.deepcopy(p2map[name])

        # prepend cost
        child.insert(0, -1)
        return child

    # ------------------------------------------------------
    # MUTATION (YOUR VERSION WITH FIXES)
    # ------------------------------------------------------
    def mutation(self, h1):
        element = copy.deepcopy(self.current_pop[h1])

        type_of_mutation = 0.85
        tasks = element[1:]  # skip cost

        if np.random.random() < type_of_mutation:
            # param mutation
            t = np.random.randint(0, len(tasks))
            task = tasks[t]

            param = np.random.randint(2, len(task))

            if param == 1:
                # toggle present flag
                task[1] = not task[1]

            else:
                name = task[0]

                if name == "MJL" or name == "Manip":
                    task[param] = np.random.random()*100

                elif name == "IK":
                    if param == 2:
                        task[param] = np.random.random()*2
                    else:
                        task[param] = np.random.random()*10

        else:
            # reorder task
            idx1, idx2 = np.random.choice(range(len(tasks)), 2, replace=False)
            tasks[idx1], tasks[idx2] = tasks[idx2], tasks[idx1]

            element[0] = -1

        return element

    # ------------------------------------------------------
    # OTHER MUTATION OPERATORS
    # ------------------------------------------------------

    def mutate_reorder(self, h1):
        element = copy.deepcopy(self.current_pop[h1])
        tasks = element[1:]

        if len(tasks) > 1:
            i, j = np.random.choice(range(len(tasks)), 2, replace=False)
            task = tasks.pop(i)
            tasks.insert(j, task)

        return element

    def mutate_insert(self, h1):
        element = copy.deepcopy(self.current_pop[h1])
        tasks = element[1:]

        new_task = self.make_random_task()
        pos = np.random.randint(0, len(tasks)+1)
        tasks.insert(pos, new_task)

        return element

    def mutate_delete(self, h1):
        element = copy.deepcopy(self.current_pop[h1])
        tasks = element[1:]

        if len(tasks) > 1:
            idx = np.random.randint(0, len(tasks))
            del tasks[idx]

        return element

    def mutate_param_swap(self, h1):
        element = copy.deepcopy(self.current_pop[h1])
        tasks = element[1:]

        idx = np.random.randint(0, len(tasks))
        task = tasks[idx]

        if len(task) > 3:
            i, j = np.random.choice(range(2, len(task)), 2, replace=False)
            task[i], task[j] = task[j], task[i]

        return element

    def mutate_noise(self, h1, scale=0.1):
        element = copy.deepcopy(self.current_pop[h1])
        tasks = element[1:]

        idx = np.random.randint(0, len(tasks))
        task = tasks[idx]

        for i in range(2, len(task)):
            if isinstance(task[i], (int, float)):
                sigma = scale * abs(task[i]) if task[i] != 0 else scale
                task[i] += np.random.normal(0, sigma)

        return element

    def mutate_replace_task(self, h1):
        element = copy.deepcopy(self.current_pop[h1])
        tasks = element[1:]

        idx = np.random.randint(0, len(tasks))
        name = tasks[idx][0]
        tasks[idx] = self.make_random_task(name)

        return element

    # ------------------------------------------------------
    # MULTIPOINT CROSSOVER & BLEND (fixed)
    # ------------------------------------------------------

    def crossover_multipoint(self, h1, h2, points=3):
        parent1 = copy.deepcopy(self.current_pop[h1])
        parent2 = copy.deepcopy(self.current_pop[h2])

        tasks1 = parent1[1:]
        tasks2 = parent2[1:]

        p2map = {t[0]: t for t in tasks2}

        n = len(tasks1)
        child = [None] * n

        switches = sorted(np.random.choice(range(n), points, replace=False))
        take_p1 = True
        next_sw = 0

        for i in range(n):
            if next_sw < len(switches) and i == switches[next_sw]:
                take_p1 = not take_p1
                next_sw += 1

            name = tasks1[i][0]

            if take_p1:
                child[i] = copy.deepcopy(tasks1[i])
            else:
                child[i] = copy.deepcopy(p2map[name])

        child.insert(0, -1)
        return child

    def crossover_blend(self, h1, h2):
        parent1 = copy.deepcopy(self.current_pop[h1])
        parent2 = copy.deepcopy(self.current_pop[h2])

        tasks1 = parent1[1:]
        tasks2 = parent2[1:]
        p2map = {t[0]: t for t in tasks2}

        child = []

        for t1 in tasks1:
            name = t1[0]
            t2 = p2map[name]

            new_task = [name]

            # present flag
            new_task.append(t1[1] if np.random.random() < 0.5 else t2[1])

            # numeric params (blend)
            for p1, p2 in zip(t1[2:], t2[2:]):
                if isinstance(p1, (int, float)):
                    alpha = np.random.random()
                    new_task.append(alpha * p1 + (1 - alpha) * p2)
                else:
                    new_task.append(p1 if np.random.random() < 0.5 else p2)

            child.append(new_task)

        child.insert(0, -1)
        return child

    # ------------------------------------------------------
    def new_population(self):
        if len(self.cache) == 0:
            print("[WARN] Cache is empty — population will be reinitialized.")
            self.current_pop = []
            for _ in range(self.size):
                self.init_pop()
            return

        # All unary genetic operators
        unary_ops = [self.mutation]
        """, self.mutate_reorder,
        self.mutate_insert,
        self.mutate_delete,
        self.mutate_param_swap,
        self.mutate_noise,
        self.mutate_replace_task]"""
        

        # All binary operators (need 2 elites)
        binary_ops = [self.crossover]
        """,
            self.crossover_multipoint,
            self.crossover_blend
        ]"""

        new_pop = []
        for elem in self.cache:
            new_pop.append(elem)

        while len(new_pop) < self.size:

            # -----------------------
            # Select base elite
            # -----------------------
            e1 = copy.deepcopy(random.choice(self.cache))
            idx1 = np.random.randint(0, len(self.cache))  # needed for mutation/crossover index
            e1[0] = -1  # reset cost

            # -----------------------
            # Choose operation
            # -----------------------
            if np.random.random() < 0.65:
                # 65% chance unary mutation
                op = np.random.choice(unary_ops)

                child = op(idx1)  # operator uses an index into current_pop or cache
                # If operator returns a bad result, fallback to elite
                if child is None:
                    child = e1

            else:
                # 35% chance crossover (requires second elite)
                op = np.random.choice(binary_ops)

                # Select second elite (different from first)
                if len(self.cache) > 1:
                    e2_idx = np.random.randint(0, len(self.cache))
                    while e2_idx == idx1:
                        e2_idx = np.random.randint(0, len(self.cache))
                else:
                    e2_idx = idx1  # if only one, fallback gracefully

                child = op(idx1, e2_idx)

                if child is None:
                    child = e1

            # Safety: guarantee cost in position 0
            if isinstance(child, list):
                child[0] = -1

            new_pop.append(child)

        # Replace population
        self.current_pop = new_pop

    def distance(self, h1, h2, w_order=1.0, w_param=1.0):
    
        sot1 = self.current_pop[h1][1:]  # skip cost
        sot2 = self.current_pop[h2][1:]

        max_len = max(len(sot1), len(sot2))
        dist_order = 0.0
        dist_param = 0.0

        # Build dicts for fast lookup
        dict1 = {t[0]: t for t in sot1}
        dict2 = {t[0]: t for t in sot2}

        # Define max values for numeric params for normalization
        param_max = {"IK": [2, 10], "Manip": [100], "MJL": [100]}

        # Extend lists to equal length (pad with None)
        sot1_ext = sot1 + [None]*(max_len - len(sot1))
        sot2_ext = sot2 + [None]*(max_len - len(sot2))

        for i, (t1, t2) in enumerate(zip(sot1_ext, sot2_ext)):
            weight = 1.0 / (i + 1)  # positional weight

            # ----- Task order mismatch -----
            name1 = t1[0] if t1 is not None else None
            name2 = t2[0] if t2 is not None else None
            if name1 != name2:
                dist_order += weight

            # ----- Parameter mismatch (only if task exists in both) -----
            if t1 is not None and t2 is not None:
                # Present flag
                flag_diff = abs(int(t1[1]) - int(t2[1]))
                dist_param += weight * flag_diff

                # Numeric parameters
                nums1 = t1[2:]
                nums2 = t2[2:]

                if name1 in param_max:
                    max_vals = param_max[name1]
                    for j, (p1, p2) in enumerate(zip(nums1, nums2)):
                        norm = max_vals[j] if j < len(max_vals) else 1.0
                        dist_param += weight * abs(p1 - p2) / norm

            else:
                # Task missing in one SOT → maximal contribution
                dist_param += weight

        # Combine
        dist = w_order * dist_order + w_param * dist_param
        return dist

    def add_in_cache(self, elem):
        self.cache.append(elem)

    def update_cache(self, elite_count=10):
        # Sort current population by cost
        sorted_pop = sorted(self.current_pop, key=lambda x: x[0])
        # Keep top-k
        self.cache = [copy.deepcopy(ind) for ind in sorted_pop[:elite_count]]

    def build_mask(self, p_active=0.5):
        n = len(self.cache)
        if n == 0:
            self.mask = np.array([], dtype=int)
            return self.mask

        # Extract costs
        costs = np.array([ind[0] for ind in self.cache], dtype=float)

        # In case all costs are equal
        if np.all(costs == costs[0]):
            probs = np.ones(n) / n
        else:
            # Lower cost -> higher probability
            # Shift and invert: better = higher
            max_cost = np.max(costs)
            probs = max_cost - costs
            probs = probs / np.sum(probs)

        # Scale probabilities to achieve desired overall density
        probs = probs * (p_active * n) / np.sum(probs)

        # Generate mask by Bernoulli sampling
        mask = np.random.rand(n) < probs
        self.mask = mask.astype(int)
        return self.mask
