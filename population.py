"""
Classes that reperesents Individuals and Populations in evolutionary algorithms.

FIXME: This module uses global variables. This should be avoided if possible.
"""
import math, random
from copy import copy
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import _is_arraylike

from .utils import (get_arity, keep_number_reasonable,
                    median_absolute_deviation)

####################
# Global Variables #
####################

#: Dict of all supported functions. Key is a string denoting the name of each
#: function. Key is the function implemented in a 'safe' way. In this context,
#: safe means that no domain errors or overflow errors will be raised and no
#: infinities will be produced as long as the arguements are within a reasonable
#: range.
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
    'abs' : lambda x: abs(x),
    'invrt' : lambda x: x * -1
}

#: A list of functions that take no arguments and produce ephemeral random
#: constants when called.
ERC_GENERATORS = [
    random.random,
    lambda : random.randint(-10, 10)
]

####################
# Global Functions #
####################

def generate_random_code(max_size, min_size=0, function_set=list(FUNCS.keys()),
    erc_generators=ERC_GENERATORS, func_constant_ratio=0.7):
    """Generates a random list of strings (represtenting functions) and floating
    constants that can be executed as a program.

    Parameters
    ----------
    max_size : int
        Maximum number of elements in the program.

    min_size : int
        Minimum number of elements in the program. Default 0.

    function_set : list[str]
        List of strings denoting which functions that could potentially be
        included in the generated program. Each string in the list must exist as
        a the key to one of the key-value pairs in FUNCS. Corisponding value
        must be a function that expects an arbirary number of floats as args (or
        no arguments). Defaults to all supported functions.

    erc_generators : list[func]
        List of funtions that produce ephemeral random constants. Ephemeral
        random constants are randomly generated constants that are set at the
        time of code generation. Default erc_generators include 2 functions: 1)
        a function that produces a float, f, such that 0 <= f < 1 and 2) a
        function that produces an integer, i, such that 0 <= i < 10.

    func_constant_ratio : float
        Floating point value such that 0 <= func_constant_ratio <= 0. Ratio of
        functions to constants to appear in the randomly generated program.
        A value of 0 produces a program of entirely constant values. A value of
        1 produces a program of entirely functions, with no constants. Defaults
        to 0.7.
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
    """Individual used in Population.

    Parameters
    ----------
    program : list
        List of function names and float constants that can be executed as a
        program.
    """

    def __init__(self, program):
        self.program = program

    def run_program(self, inputs=None, print_trace=False):
        """Individual used in Population.

        Parameters
        ----------
        inputs : array-like, shape = (n_features,)
            List of values for each feature that can be accessed by the input_n
            functions found in the program.

        print_trace : bool
            If true, prints the current program element and the state of the
            stack at each step of executing the program. Defaults to False.
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
        """Evaluates the individual.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        y : {array-like, sparse matrix}, shape = (n_samples, 1)
            Labels.

        metric : function
            Function to used to calculate the error of the individual. All
            sklearn regression metrics are supported.
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
        """Simplifies the individual's program by randomly removing some
        elements of the program and confirming that the total error remains the
        same or lower.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        y : {array-like, sparse matrix}, shape = (n_samples, 1)
            Target values.

        metric : function
            Function to used to calculate the error of an individual. All
            sklearn regression metrics are supported.

        steps : int
            Number of simplification iterations to perform.
        """
        for step in range(steps):
            # Record the origional error and program
            original_error = copy(self.total_error)
            original_prog = self.program[:]
            # Pick random element from program to remove.
            num_rm = random.randint(1, 3)
            for i in range(num_rm):
                rm_ind = random.randint(0, len(self.program)-1)
                del self.program[rm_ind]
            # Evaluate the new program.
            self.evaluate(X, y, metric)
            # If program performance gets worse, put back the old program.
            if self.total_error > original_error:
                self.program = original_prog
                self.evaluate(X, y, metric)

    def __repr__(self):
        return '<Individual: ' + str(self.program) + '>'

class Population(list):
    """
    Symbolic regression population.
    """

    def evaluate(self, X, y, metric=mean_squared_error):
        """Evaluates every individual in the population, if the individual has
        not been previously evaluated.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        y : {array-like, sparse matrix}, shape = (n_samples, 1)
            Target values.

        metric : function
            Function to used to calculate the error of an individual. All
            sklearn regression metrics are supported.
        """
        for i in self:
            if not hasattr(i, 'error_vector'):
                i.evaluate(X, y, metric)

    def lexicase_selection(self):
        """Returns an individual that does the best on the fitness cases when
        considered one at a time in random order.

        http://faculty.hampshire.edu/lspector/pubs/lexicase-IEEE-TEC.pdf

        Returns
        -------
        individual : Individual
            An individual from the population selected using lexicase selection.
        """
        candidates = self[:]
        cases = list(range(len(self[0].error_vector)))
        random.shuffle(cases)
        while len(cases) > 0 and len(candidates) > 1:
            best_val_for_case = min([ind.error_vector[cases[0]] for ind in candidates])
            candidates = [ind for ind in candidates if ind.error_vector[cases[0]] == best_val_for_case]
            cases.pop(0)
        return random.choice(candidates)

    def epsilon_lexicase_selection(self, epsilon='auto'):
            """Returns an individual that does the best on the fitness cases
            when considered one at a time in random order.

            https://push-language.hampshire.edu/uploads/default/original/1X/35c30e47ef6323a0a949402914453f277fb1b5b0.pdf

            Parameters
            ----------
            epsilon : {'auto', float, array-like}
                If an individual is within epsilon of being elite, it will
                remain in the selection pool. If 'auto', epsilon is set at
                the start of each selection even to be equal to the
                Median Absolute Deviation of each test case.

            Returns
            -------
            individual : Individual
                An individual from the population selected using lexicase
                selection.
            """
            candidates = self[:]
            cases = list(range(len(self[0].error_vector)))
            random.shuffle(cases)

            if epsilon == 'auto':
                all_errors = np.array([i.error_vector[:] for i in candidates])
                epsilon = np.apply_along_axis(median_absolute_deviation, 0,
                                              all_errors)

            while len(cases) > 0 and len(candidates) > 1:
                case = cases[0]
                errors_this_case = [i.error_vector[case] for i in candidates]
                best_val_for_case = min(errors_this_case)
                if _is_arraylike(epsilon):
                    max_error = best_val_for_case + epsilon[case]
                else:
                    max_error = best_val_for_case + epsilon
                test = lambda i: i.error_vector[case] <= max_error
                candidates = [i for i in candidates if test(i)]
                cases.pop(0)
            return random.choice(candidates)

    def tournament_selection(self, tournament_size=7):
        """Returns the individual with the lowest error within a random
        tournament.

        Parameters
        ----------
        tournament_size : int
            Size of each tournament.

        Returns
        -------
        individual : Individual
            An individual from the population selected using tournament
            selection.
        """
        tournament = []
        for _ in range(tournament_size):
            tournament.append(random.choice(self[:]))
        min_error_in_tourn = min([ind.total_error for ind in tournament])
        best_in_tourn = [ind for ind in tournament if ind.total_error == min_error_in_tourn]
        return best_in_tourn[0]

    def select(self, method='epsilon_lexicase', epsilon='auto',
        tournament_size=7):
        """Selects a individual from

        Parameters
        ----------
        method : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        y : {array-like, sparse matrix}, shape = (n_samples, 1)
            Labels.


        """
        if method == 'epsilon_lexicase':
            return self.epsilon_lexicase_selection(epsilon)
        elif method == 'lexicase':
            return self.lexicase_selection()
        elif method == 'tournament':
            return self.tournament_selection(tournament_selection)
        else:
            raise ValueError("Unknown selection method: " + str(method))

    def lowest_error(self):
        """Returns the lowest total error found in the population.
        """
        gnrtr = (ind.total_error for ind in self)
        return np.min(np.fromiter(gnrtr, np.float))

    def average_error(self):
        """Returns the average total error found in the population.
        """
        gnrtr = (ind.total_error for ind in self)
        return np.mean(np.fromiter(gnrtr, np.float))

    def unique(self):
        """Returns the number of unique programs found in the population.
        """
        programs_set = {str(ind.program[:]) for ind in self}
        return len(programs_set)

# pop = Population()
# for i in range(10):
#     pop.append(Individual(generate_random_code(40)))
# print(pop)
# print()
# for i in pop:
#     print(i.run_program())

# print(generate_random_code(1, min_size=1))
