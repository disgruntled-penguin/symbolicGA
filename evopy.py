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
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.expression = self.mk_rand_exp(depth=0)
        self.func = self.compile_fn()
        
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


if __name__ == "__main__":

    ind = Individual(max_depth=5)
    filename=f'{ind.expression}.png'
    ind.render_image(filename, grid_size=500)
    
    print(f"Generated expression: {ind.expression}")