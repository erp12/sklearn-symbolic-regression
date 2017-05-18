"""
TODO
"""

import numpy as np
from numpy.random import choice
import random

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import validation, check_array
from sklearn.metrics import mean_squared_error
from sklearn.utils import validation, check_array

from .population import FUNCS, ERC_GENERATORS, generate_random_code, Individual, Population
from .variation import UniformMutator, Alternator

##############
# Estimators #
##############

DEFAULT_OPERATORS = [
    (UniformMutator(), 0.3),
    (Alternator(), 0.7)
]


class SymbolicRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, population_size=100, max_generations=200,
                 selection_method='epsilon_lexicase',
                 max_initial_program_size=50, max_program_size=300,
                 epsilon=None, tournament_size=7, fit_metric=mean_squared_error,
                 function_set=list(FUNCS.keys()), erc_generators=ERC_GENERATORS,
                 operators=DEFAULT_OPERATORS, simplification_steps=500,
                 verbose=0):
        """ToDo
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.selection_method = selection_method
        self.max_initial_program_size = max_initial_program_size
        self.max_program_size = max_program_size
        self.epsilon = epsilon
        self.tournament_size = tournament_size
        self.fit_metric = fit_metric
        self.function_set = function_set
        self.erc_generators = erc_generators
        self.operators = operators
        self.simplification_steps = simplification_steps
        self.verbose = verbose

    def fit(self, X, y):
        """TODO: Write Docstring
        TODO: Add verbose printing
        """
        # TODO: Add some validation checks
        n_feats = X.shape[1]
        all_funcs = self.function_set + \
            ['input_' + str(i) for i in range(n_feats)]

        pop = Population()
        for i in range(self.population_size):
            pop.append(Individual(generate_random_code(
                self.max_initial_program_size)))
        # Evaluate initial population.
        pop.evaluate(X, y, self.fit_metric)

        for g in range(self.max_generations):
            next_gen = Population()
            for i in range(self.population_size):
                op = choice(
                    [o[0] for o in self.operators],
                    1,
                    [o[1] for o in self.operators]
                )[0]

                if op.operator_type is 'mutation':
                    p = pop.select(self.selection_method,
                                   epsilon=self.epsilon,
                                   tournament_size=self.tournament_size)
                    next_gen.append(op.produce(p, function_set=all_funcs))
                elif op.operator_type is 'recombination':
                    p1 = pop.select(self.selection_method,
                                    epsilon=self.epsilon,
                                    tournament_size=self.tournament_size)
                    p2 = pop.select(self.selection_method,
                                    epsilon=self.epsilon,
                                    tournament_size=self.tournament_size)
                    next_gen.append(op.produce(p1, p2))

            pop = next_gen
            # Evaluation
            pop.evaluate(X, y, self.fit_metric)

        self.best_error_ = min([i.total_error for i in pop])
        self.best_ = [i for i in pop if i.total_error == self.best_error_][0]

        if self.verbose > 0:
            print("Final program size:", len(self.best_.program))

        self.best_.simplify(X, y, self.fit_metric, self.simplification_steps)

        if self.verbose > 0:
            print("Simplified program size:", len(self.best_.program))

    def predict(self, X):
        """TODO
        """
        validation.check_is_fitted(self, 'best_')
        return np.apply_along_axis(self.best_.run_program, 1, X)
