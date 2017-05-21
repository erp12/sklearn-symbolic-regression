"""
Symbolic regression

FIXME: This module uses a global variable. This should be avoided if possible.
"""

import numpy as np
from numpy.random import choice
import random

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import validation, check_array
from sklearn.metrics import mean_squared_error
from sklearn.utils import validation, check_array

from .population import (FUNCS, ERC_GENERATORS, generate_random_code,
                         Individual, Population)
from .variation import UniformMutator, Alternator

##############
# Estimators #
##############

DEFAULT_OPERATORS = [
    (UniformMutator(), 0.3),
    (Alternator(), 0.7)
]


class SymbolicRegressor(BaseEstimator, RegressorMixin):
    """
    Symbolic regressor.

    TODO: Implement n_jobs for evaluation.

    Parameters
    ----------
    population_size : int
        Number of individuals in the population at each generation.

    max_generations : int
        Number of generations to run evolution.

    selection_method : str
        Name of algorithm to use when selecting parents for variation operators.
        Options are 'epsilon_lexicase', 'lexicase', and 'tournament'.

    max_initial_program_size : int
        Maximium number of elements that can appear in any randomly generated
        program.

    max_program_size : int
        Maximum size a program can be at any point during evolution.

    epsilon : None, float
        When selection method is 'epsilon_lexicase', this value is used as
        epsilon. When set to 'auto', epsilon is calculated using automatically
        using the MAD. Default is 'auto'.

    tournament_size : int
        When selection method is `tournament`, this value controls the size of
        each tournament. Default is 7.

    fit_metric : func
        A function (probably one of sklearn's regression scoring functions)
        that returns an overal score for the SymbolicRegressor. This is used
        to determine the total error of an Individual during evoltion. Defaults
        to mean_squared_error.

    function_set : list
        A list of strings where each string is the name of a supported function
        that can appear in an evolved program. Defaults to all supported
        functions.

    erc_generators : list
        A list of functions where each function can be called with no arguments
        and returns a random number. These functions are also known as
        "ephemeral random constant generators". Numbers returned by these
        functions are also

    operators : list
        A list containing tuples. Each tuple contains 2 elements: 1) an
        instance of one of the VariationOperator subclasses and 2) a float. The
        float denotes how frequently (relative to the other floats) the operator
        is performed to create a new child in the next generations. For example,
        if operators looks like [(A(), 0.2), (B(), 0.8)] then operator A will
        be used to create roughly 1/5 of the children in each generation, while
        operator B will create roughly 4/5. Defaults to DEFAULT_OPERATORS list
        stored in a global variable.

    simplification_steps : int
        After after evolution has reached it's max generations, programs can be
        simplified by removing random elements and confimrming the error has
        not gotten worse. It has been shown that this can  improve
        generalization in some situations, but it also improves program
        readability and slightly improves prediction time. Defaults to 500.

    verbose : int
        When values is greater than 0, verbose printing is enabled.

    Attributes
    ----------

    best_ : Individual
        Best Individual present in the last generation of evolution.

    best_error_ : float
        Total error of the Individual stored in best_. This is considered the
        overall training error of the SymbolicRegressor.
    """

    def __init__(self, population_size=100, generations=200,
                 selection_method='epsilon_lexicase',
                 max_initial_program_size=50, max_program_size=300,
                 epsilon='auto', tournament_size=7,
                 fit_metric=mean_squared_error, function_set=list(FUNCS.keys()),
                 erc_generators=ERC_GENERATORS, operators=DEFAULT_OPERATORS,
                 simplification_steps=500, verbose=0):
        self.population_size = population_size
        self.generations = generations
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
        """Fits the SymbolicRegressor.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        y : {array-like, sparse matrix}, shape = (n_samples, 1)
            Target values.
        """
        n_feats = X.shape[1]
        all_funcs = self.function_set + \
            ['input_' + str(i) for i in range(n_feats)]

        pop = Population()
        for i in range(self.population_size):
            pop.append(Individual(generate_random_code(
                self.max_initial_program_size)))
        # Evaluate initial population.
        pop.evaluate(X, y, self.fit_metric)

        for g in range(self.generations):

            if self.verbose > 0:
                print('Generation:', g,
                      '| Lowest Error:', pop.lowest_error(),
                      '| Avg Error:', pop.average_error(),
                      '| Number of Unique Programs:', pop.unique())

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
        """Predict using the best program found by evolution.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        validation.check_is_fitted(self, 'best_')
        return np.apply_along_axis(self.best_.run_program, 1, X)
