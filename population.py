"""
TODO: Write modeule docstring
"""
import math, random
from copy import copy
import numpy as np

from sklearn.metrics import mean_squared_error

from .utils import get_arity, keep_number_reasonable

####################
# Global Variables #
####################

# Stack Based Programs
FUNCS = {
    '+' : lambda x,y: y + x,
    '-' : lambda x,y: y - x,
    '*' : lambda x,y: y * x,
    '/' : lambda x,y: None if x == 0 else y / x,
    '^2' : lambda x: x * x,
    '^3' : lambda x: x * x * x,
    #'^x' : lambda x,y: y ** x,
    'sqrt' : lambda x: None if x <= 0 else math.sqrt(x),
    'log' : lambda x: None if x <= 0 else math.log(x),
    #'log_b' : lambda x, y: None if x <= 0 or y <= 0 else math.log(y, x),
    'sin' : lambda x: math.sin(x),
    'cos' : lambda x: math.cos(x),
    'abs' : lambda x: abs(x)
}

ERC_GENERATORS = [
    random.random,
    lambda : random.randint(-10, 10)
]

####################
# Global Functions #
####################

def generate_random_code(max_size, min_size=0, function_set=list(FUNCS.keys()),
    func_constant_ratio=0.7, erc_generators=ERC_GENERATORS):
    """TODO: Write docstring
    """
    size = random.randint(min_size, max_size)
    prog = []
    for i in range(size):
        if random.random() < func_constant_ratio:
            prog.append(random.choice(function_set))
        else:
            erc_gen = random.choice(erc_generators)
            prog.append(erc_gen())
    return prog

###########
# Classes #
###########

class Individual:
    """TODO: Write docstring
    """

    def __init__(self, program):
        self.program = program

    def run_program(self, inputs=None, print_trace=False):
        """TODO: Write docstring
        """
        if print_trace:
            print('Begin Program Execuation')

        stack = []
        for el in self.program:
            if print_trace:
                print("Next:", el)
                print('Stack:', stack)
                print()
            if isinstance(el, str):
                if 'input_' in el:
                    # Element denotes input.
                    i = int(el[6:])
                    stack.append(inputs[i])
                else:
                    f = FUNCS[el]
                    arity = get_arity(f)
                    # Ensure enough arguments exist on stacks.
                    if arity <= len(stack):
                        # Get arguments
                        args = stack[-arity:]
                        # Get output of instruction
                        output = f(*args)
                        # Check that f returns valid result.
                        if not output is None:
                            # Pop arguments off
                            for a in range(arity):
                                stack.pop()
                            # Push result back onto stack.
                            stack.append(keep_number_reasonable(output))
            else:
                stack.append(float(el))
        # If stack is empty, return invalid token.
        if len(stack) == 0:
            return 'EMPTY_STACK'
        # Return output of program.
        return stack.pop()

    def evaluate(self, X, y, metric=mean_squared_error):
        """TODO: Write Docstring
        TODO: Check X is 2D.
        TODO: Warn that only regression metrics work.
        """
        y_hat = []
        error_vec = []
        row_index = 0
        for r in X:
            out = self.run_program(r)
            if out is 'EMPTY_STACK':
                y_hat.append(0)
                error_vec.append(9999)
            else:
                y_hat.append(out)
                error_vec.append(abs(y[row_index] - out))
            row_index += 1
        self.error_vector = error_vec
        self.total_error = metric(y, y_hat)

    def simplify(self, X, y, metric=mean_squared_error, steps=500):
        """TODO: Write docstring.
        TODO: Check evaluated.
        """
        for step in range(steps):
            # Record the origional error and program
            original_error = copy(self.total_error)
            original_prog = self.program[:]
            # Pick random element from program to remove.
            rm_ind = random.randint(0, len(self.program))
            new_program = self.program[:rm_ind] + self.program[rm_ind+1:]
            # Evaluate the new program.
            self.program = new_program
            self.evaluate(X, y, metric)
            # If program performance gets worse, put back the old program.
            if self.total_error > original_error:
                self.program = original_prog
                self.evaluate(X, y, metric)

    def __repr__(self):
        return '<Individual: ' + str(self.program) + '>'

class Population(list):
    """TODO: Write docstring
    """

    def evaluate(self, X, y, metric=mean_squared_error):
        """TODO: Write docstring
        """
        for i in self:
            if not hasattr(i, 'error_vector'):
                i.evaluate(X, y, metric)

    def lexicase_selection(self):
        """Returns an individual that does the best on the fitness cases when
        considered one at a time in random order.

        http://faculty.hampshire.edu/lspector/pubs/lexicase-IEEE-TEC.pdf

        :returns: A selected individual.
        """
        candidates = self[:]
        cases = list(range(len(self[0].error_vector)))
        random.shuffle(cases)
        while len(cases) > 0 and len(candidates) > 1:
            best_val_for_case = min([ind.error_vector[cases[0]] for ind in candidates])
            candidates = [ind for ind in candidates if ind.error_vector[cases[0]] == best_val_for_case]
            cases.pop(0)
        return random.choice(candidates)

    def epsilon_lexicase_selection(self, epsilon=None):
            """Returns an individual that does the best on the fitness cases when
            considered one at a time in random order. Requires a epsilon parameter.

            https://push-language.hampshire.edu/uploads/default/original/1X/35c30e47ef6323a0a949402914453f277fb1b5b0.pdf

            .. todo::
                Adjust this implementation based on recent finding with epsilon lexicase
                (ie. static, dynamic, super-dynamic, etc)

            :param float epsilon: If an individual is within epsilon of being elite, it will \
            remain in the selection pool. If 'dynamic', epsilon is set at the start of \
            each selection even. If 'super-dynamic', epsilon is set realtive to the \
            current selection pool at each iteration of lexicase selection.
            :returns: A list of selected individuals.
            """
            candidates = self[:]
            cases = list(range(len(self[0].error_vector)))
            random.shuffle(cases)
            while len(cases) > 0 and len(candidates) > 1:
                errors_for_this_case = [ind.error_vector[cases[0]] for ind in candidates]
                best_val_for_case = min(errors_for_this_case)
                if epsilon == None:
                    median_val = np.median(errors_for_this_case)
                    median_absolute_deviation = np.median([abs(x - median_val) for x in errors_for_this_case])
                    epsilon = median_absolute_deviation
                candidates = [ind for ind in candidates if ind.error_vector[cases[0]] <= best_val_for_case + epsilon]
                cases.pop(0)
            return random.choice(candidates)

    def tournament_selection(self, tournament_size=7):
        """Returns k individuals that do the best out of their respective
        tournament.

        :param int tournament_size: Size of each tournament.
        :returns: A list of selected individuals.
        """
        tournament = []
        for _ in range(tournament_size):
            tournament.append(random.choice(self[:]))
        min_error_in_tourn = min([ind.total_error for ind in tournament])
        best_in_tourn = [ind for ind in tournament if ind.total_error == min_error_in_tourn]
        return best_in_tourn[0]

    def select(self, method='epsilon_lexicase', epsilon=None,
        tournament_size=7):
        """TODO: Write Docstring
        """
        if method == 'epsilon_lexicase':
            return self.epsilon_lexicase_selection(epsilon)
        elif method == 'lexicase':
            return self.lexicase_selection()
        elif method == 'tournament':
            return self.tournament_selection(tournament_selection)
        else:
            raise ValueError("Unknown selection method: " + str(method))

# pop = Population()
# for i in range(10):
#     pop.append(Individual(generate_random_code(40)))
# print(pop)
# print()
# for i in pop:
#     print(i.run_program())

# print(generate_random_code(1, min_size=1))
