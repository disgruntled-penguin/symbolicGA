import random
import numpy as np
import sympy
from sympy.utilities.lambdify import lambdify
import matplotlib.pyplot as plt

#genetic bb
x, y, r, theta = sympy.symbols('x y r theta')

#leaves of the expression tree
TERMINALS = [x, y, r, theta, sympy.Integer(1), sympy.Integer(2)]


def safe_log(a):
    return sympy.log(sympy.Max(a, 1e-6))

def safe_mod(a, b):
    epsilon = 1e-6
    safe_b = sympy.Max(sympy.Abs(b), epsilon)
    return a % safe_b

#branches of the expression tree
OPERATORS = [
    (sympy.Add, 2),
    (sympy.Mul, 2),
    (sympy.sin, 1),
    (sympy.cos, 1),
    (sympy.exp, 1),
    (safe_log, 1),
    (sympy.tanh, 1),  
    (sympy.Abs, 1),   
    (safe_mod, 2),    
]

class Individual:
    def __init__(self, max_depth=4, expr_r=None, expr_g=None, expr_b=None):
        self.max_depth = max_depth
      
        self.expression_r = expr_r if expr_r else self.mk_rand_exp(depth=0)
        self.expression_g = expr_g if expr_g else self.mk_rand_exp(depth=0)
        self.expression_b = expr_b if expr_b else self.mk_rand_exp(depth=0)

        self.func_r = self.compile_fn(self.expression_r)
        self.func_g = self.compile_fn(self.expression_g)
        self.func_b = self.compile_fn(self.expression_b)

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

    def compile_fn(self, expression_to_compile): 
        try:
            return lambdify([x, y, r, theta], expression_to_compile, 'numpy') 
        except Exception as e:
            return lambdify([x, y, r, theta], x, 'numpy')
    
    def evaluate(self, grid_size=300):
      lin = np.linspace(-5, 5, grid_size)
      xx, yy = np.meshgrid(lin, lin)
      rr = np.sqrt(xx**2 + yy**2)
      tt = np.arctan2(yy, xx)

      with np.errstate(divide='ignore', invalid='ignore'):
        z_r = self.func_r(xx, yy, rr, tt)
        z_g = self.func_g(xx, yy, rr, tt)
        z_b = self.func_b(xx, yy, rr, tt)

        def process_channel(z):
         if not isinstance(z, np.ndarray) or z.ndim == 0:
            z = np.full_like(xx, z, dtype=np.float64)
         if np.iscomplexobj(z):
            z = np.real(z)
         z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
         if z.max() != z.min():
            z = (z - z.min()) / (z.max() - z.min())
         else:
            z = np.zeros_like(z)
         return z

      r_channel = process_channel(z_r)
      g_channel = process_channel(z_g)
      b_channel = process_channel(z_b)

      rgb_image = np.stack([r_channel, g_channel, b_channel], axis=-1)
      return rgb_image

    def render_image(self, filename="output.png", grid_size=300):
      rgb_image = self.evaluate(grid_size)
      plt.imsave(filename, rgb_image, origin='lower')

class Population: #evol loop
    def __init__(self, size=50, max_depth=5):
        self.size = size
        self.max_depth = max_depth
        self.generation = 0
        self.individuals = [Individual(max_depth) for _ in range(size)]

    def calculate_fitness(self):

        epsilon = 1e-6 
        
        for ind in self.individuals:
            rgb_grid = ind.evaluate(grid_size=100)

            r_channel = rgb_grid[..., 0]
            g_channel = rgb_grid[..., 1]
            b_channel = rgb_grid[..., 2]
            
            total_fitness = 0

            for channel in [r_channel, g_channel, b_channel]:
                score_spread = np.std(channel)
                
                g_x, g_y = np.gradient(channel)
                score_edges = np.mean(g_x**2 + g_y**2)
            
                balanced_score = (score_spread + epsilon) * (score_edges + epsilon)
                total_fitness += balanced_score
            
            ind.fitness = total_fitness
            
            if ind.fitness < 1e-10: 
                ind.fitness = 1e-10



    def select_parents(self):
        parents = []
        for _ in range(2): #picks 3 random
            tournament = random.sample(self.individuals, 3)
  
            winner = max(tournament, key=lambda ind: ind.fitness)
            parents.append(winner)
        return parents[0], parents[1]

    def _crossover_channel(self, expr1, expr2):
        nodes1 = list(sympy.preorder_traversal(expr1))
        nodes2 = list(sympy.preorder_traversal(expr2))

        if len(nodes1) > 1 and len(nodes2) > 1:
            crossover_point1 = random.choice(nodes1[1:])
            crossover_point2 = random.choice(nodes2[1:])
            try:
         
                return expr1.subs(crossover_point1, crossover_point2)
            except Exception:
                return expr1 
        else:
            return expr1 

    def crossover(self, parent1, parent2):

        child_expr_r = self._crossover_channel(parent1.expression_r, parent2.expression_r)
        child_expr_g = self._crossover_channel(parent1.expression_g, parent2.expression_g)
        child_expr_b = self._crossover_channel(parent1.expression_b, parent2.expression_b)

        return Individual(
            self.max_depth, 
            expr_r=child_expr_r, 
            expr_g=child_expr_g, 
            expr_b=child_expr_b
        )

    def mutate(self, individual):

        channel = random.choice([1, 2, 3])
        
        if channel == 1:
            expr_to_mutate = individual.expression_r
        elif channel == 2:
            expr_to_mutate = individual.expression_g
        else:
            expr_to_mutate = individual.expression_b


        nodes = list(sympy.preorder_traversal(expr_to_mutate))
        if len(nodes) < 2:
            return # Too simple
            
        node_to_mutate = random.choice(nodes[1:]) 
        new_subtree = individual.mk_rand_exp(depth=individual.max_depth - 1)
 
        try:
            mutated_expr = expr_to_mutate.subs(node_to_mutate, new_subtree)
        except Exception:
            mutated_expr = expr_to_mutate 


        if channel == 1:
            individual.expression_r = mutated_expr
            individual.func_r = individual.compile_fn(mutated_expr)
        elif channel == 2:
            individual.expression_g = mutated_expr
            individual.func_g = individual.compile_fn(mutated_expr)
        else:
            individual.expression_b = mutated_expr
            individual.func_b = individual.compile_fn(mutated_expr)


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
        
        print(f" gen{i} - fitness/expression: {best_ind.fitness:.4f}/{best_ind.expression_r}")

