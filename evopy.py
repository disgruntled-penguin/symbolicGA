import random
import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

#genetic bb
x, y = sympy.symbols('x y')

#leaves of the expression tree
TERMINALS = [x, y, sympy.Integer(1), sympy.Integer(2), sympy.Integer(3)]


def safe_log(a):
    return sympy.log(sympy.Max(a, 1e-6))

#branches of the expression tree
OPERATORS = [
    (sympy.Add, 2),
    (sympy.Mul, 2),
    (sympy.sin, 1),
    (sympy.cos, 1),
    (sympy.exp, 1),
    (safe_log, 1), 
]

class Individual:
    def __init__(self, max_depth=4, expression=None):
        self.max_depth = max_depth
        if expression is None:
            self.expression = self.mk_rand_exp(depth=0)
        else:
            self.expression = expression
        self.func = self.compile_fn()
        self.fitness = 0
        
    def mk_rand_exp(self, depth): # make random expression
        if depth >= self.max_depth:
            return random.choice(TERMINALS)

        if random.random() < 0.7: # 0.7 to pick an operator/grow the tree
            op, arity = random.choice(OPERATORS)
            args = [self.mk_rand_exp(depth + 1) for _ in range(arity)]
            return op(*args)
        else: # 0.3 to pick terminal/stop growing
            return random.choice(TERMINALS)

    def compile_fn(self):
       # print(f"Compiling expression: {self.expression}")
        try:
            return lambdify([x, y], self.expression, 'numpy')
        except Exception as e:
            print(f"Error compiling {self.expression}: {e}")
            return lambdify([x, y], x, 'numpy')

    def evaluate(self, grid_size=300):
        lin = np.linspace(-5, 5, grid_size) #for eg
        xx, yy = np.meshgrid(lin, lin)
        with np.errstate(divide='ignore', invalid='ignore'): 
            z_grid = self.func(xx, yy)
        
        if not isinstance(z_grid, np.ndarray) or z_grid.ndim == 0: #return only 2D array
            z_grid = np.full_like(xx, z_grid, dtype=np.float64)

        if np.iscomplexobj(z_grid):
            z_grid = np.real(z_grid) #only considers real part of complex
            
        z_grid = np.nan_to_num(z_grid, nan=0.0, posinf=0.0, neginf=0.0)
        
        # normalization
        if z_grid.max() != z_grid.min():
            z_grid = (z_grid - z_grid.min()) / (z_grid.max() - z_grid.min())
        
        return z_grid

    def render_image(self, filename="output.png", grid_size=300):
        z_grid = self.evaluate(grid_size)
        plt.imsave(filename, z_grid, cmap='viridis', origin='lower')

class Population: #evol loop
    def __init__(self, size=50, max_depth=5):
        self.size = size
        self.max_depth = max_depth
        self.generation = 0
        self.individuals = [Individual(max_depth) for _ in range(size)]

    def calculate_fitness(self): #mean of sq gradients 
        for ind in self.individuals:
            z_grid = ind.evaluate(grid_size=100) 
            g_x, g_y = np.gradient(z_grid)
        
            ind.fitness = np.mean(g_x**2 + g_y**2)
            
            if ind.fitness < 1e-5:
                ind.fitness = 1e-5

    def select_parents(self):
        parents = []
        for _ in range(2): #picks 3 random
            tournament = random.sample(self.individuals, 3)
  
            winner = max(tournament, key=lambda ind: ind.fitness)
            parents.append(winner)
        return parents[0], parents[1]

    def crossover(self, parent1, parent2):

        nodes1 = list(sympy.preorder_traversal(parent1.expression))
        nodes2 = list(sympy.preorder_traversal(parent2.expression))

        if len(nodes1) > 1 and len(nodes2) > 1:
            crossover_point1 = random.choice(nodes1[1:])
            crossover_point2 = random.choice(nodes2[1:])

            try:
                child_expr = parent1.expression.subs(crossover_point1, crossover_point2)
            except Exception:
                child_expr = parent1.expression
        else: #just x
            child_expr = parent1.expression 

        return Individual(self.max_depth, expression=child_expr)

    def mutate(self, individual):
        nodes = list(sympy.preorder_traversal(individual.expression))
        if len(nodes) < 2:
            return #too simple to mutate
            
        node_to_mutate = random.choice(nodes[1:]) 
        new_subtree = individual.mk_rand_exp(depth=individual.max_depth - 1)
 
        try:
            mutated_expr = individual.expression.subs(node_to_mutate, new_subtree)
        except Exception:
            mutated_expr = individual.expression 

        individual.__init__(individual.max_depth, expression=mutated_expr)


    def evolve(self):
        
        self.generation += 1
        self.calculate_fitness()
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        
        new_population = []
        elitism_count = int(self.size * 0.1)
        new_population.extend(self.individuals[:elitism_count])
        
        fill_count = self.size - elitism_count
        for _ in range(fill_count):
            p1, p2 = self.select_parents()
            child = self.crossover(p1, p2)
            
            if random.random() < 0.15:
                self.mutate(child)
                
            new_population.append(child)
            
        self.individuals = new_population
        
    def best_individual(self):
        return self.individuals[0]


if __name__ == "__main__":

    NUM_GENERATIONS = 10
    POPULATION_SIZE = 50
    MAX_DEPTH = 6

    pop = Population(size=POPULATION_SIZE, max_depth=MAX_DEPTH)

    for i in range(NUM_GENERATIONS):
        pop.evolve()
        best_ind = pop.best_individual()
        filename = f"results/generation_{str(i).zfill(3)}.png"
        best_ind.render_image(filename=filename, grid_size=500)
        
        print(f" gen{i} - fitness/expression: {best_ind.fitness:.4f}/{best_ind.expression}")

