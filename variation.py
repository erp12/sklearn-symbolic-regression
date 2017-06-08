"""
Variation operators for genetic algorithms.
"""
from abc import ABCMeta, abstractmethod
import random

from .population import (FUNCS, ERC_GENERATORS, generate_random_code,
                         Individual, Population)
from .utils import get_arity, noise_factor

class VariationOperator(metaclass=ABCMeta):
    """Variation operator base class. All variation operators must inherit from
    this class.

    Properties
    ----------
    operator_type : str
        Denotes the type of VariationOperator. Supported options are 'mutation',
        'recombination', and 'pipeline'.

        Mutation operators require a single Individual and a function set to be
        performed.

        Recombination operators require two Individuals and no function set.

        Pipeline operators can be comprised of many other VariationOperators
        chained together and thus require two Individuals and a function set to
        be performed.

        Clone operators only require a single Individual.
    """

    _operator_type = None

    @property
    def operator_type(self):
        return self._operator_type

    @operator_type.setter
    def operator_type(self, value):
        if value not in ('mutation', 'recombination', 'pipeline', 'clone'):
            raise AttributeError('Unknown variation operator type' + str(value))
        self._operator_type = value

    @operator_type.deleter
    def operator_type(self):
        del self._operator_type

    @abstractmethod
    def produce(self):
        """Produces a child.
        """

class UniformMutator(VariationOperator):
    """Uniform Mutation operator.

    Parameters
    ----------
    rate : float
        The probablility of mutating any given element of the individual's
        program. Must be 0 <= rate <= 1. Defaults to 0.1.

    constant_perturb_rate : float
        When mutating a constant value, this is the probably of the value being
        perturbed with Gaussian noise, rather than than replaced with an
        entirely new function name or constant. Defaults to 0.5.

    perturb_standard_deviation : float
        When constant value is being perturbed with Gaussian noise, this is used
        as the standard deviation of the noise. Defaults to 1.0.
    """

    def __init__(self, rate=0.05, constant_perturb_rate=0.5,
        perturb_standard_deviation=1.0):
        self.rate = rate
        self.constant_perturb_rate = constant_perturb_rate
        self.perturb_standard_deviation = perturb_standard_deviation
        self.operator_type = 'mutation'

    def produce(self, individual, function_set):
        """Produces a child using Umiform Mutation.

        FIXME: Mutation ERC_GENERATORS cannot currently be set.

        Parameters
        ----------
        individual : Individual
            Individual whose program will be uniformly mutated to produce a
            child Individual.

        function_set : list[]
            List of function names (strings) to use when mutation overwrites
            code in an individual's program with new code. This should included
            supported function names and input_n where n is the index of a
            feature.

        Returns
        -------
        child : Individual
            A new individual.
        """
        new_prog = []
        for el in individual.program[:]:
            if random.random() > self.rate:
                new_prog.append(el)
            else:
                if isinstance(el, str) or random.random() > self.constant_perturb_rate:
                    new_prog = (new_prog +
                        generate_random_code(1, min_size=1, function_set=function_set))
                else:
                    new_const = self.perturb_standard_deviation * noise_factor() + el
                    new_prog.append(new_const)
        return Individual(new_prog)

    def __repr__(self):
        return ('<UniformMutator: ' + str(self.rate) + ', ' +
                    str(self.constant_perturb_rate) + ', ' +
                    str(self.perturb_standard_deviation) + '>')

class Alternator(VariationOperator):
    """Uniformly alternates between the two parents.

    Parameters
    ----------
    rate : float
        The probablility of switching which parent program elements are being
        copied from. Must be 0 <= rate <= 1. Defaults to 0.1.

    alignment_deviation : int
        The standard deviation of how far alternation may jump between indices
        when switching between parents.
    """

    def __init__(self, rate=0.1, alignment_deviation=10):
        self.rate = rate
        self.alignment_deviation = alignment_deviation
        self.operator_type = 'recombination'

    def produce(self, individual1, individual2):
        """Produces a child using the alternation operator.

        Parameters
        ----------
        individual1 : Individual
            The first Individual whose program will be used during alternation
            to produce a child program.

        individual2 : Individual
            The second Individual whose program will be used during alternation
            to produce a child program.

        Returns
        -------
        child : Individual
            A new individual.
        """
        parent_1 = individual1.program
        parent_2 = individual2.program
        new_prog = []
        # Random pick which parent to start from
        use_parent_1 = random.choice([True, False])
        loop_times = len(parent_1)
        if not use_parent_1:
            loop_times = len(parent_2)

        i = 0
        while (i < loop_times):
            if random.random() < self.rate:
                # Switch which parent we are pulling genes from
                i += round(self.alignment_deviation * noise_factor())
                i = int(max(0, i))
                use_parent_1 = not use_parent_1
            else:
                # Pull gene from parent
                if use_parent_1:
                    new_prog.append(parent_1[i])
                else:
                    new_prog.append(parent_2[i])
                i = int(i+1)

            # Change loop stop condition
            loop_times = len(parent_1)
            if not use_parent_1:
                loop_times = len(parent_2)

        return Individual(new_prog)

        def __repr__(self):
            return ('<Alternator: ' + str(self.rate) + ', ' +
                        str(self.alignment_deviation) + '>')

class OperatorPipeline(VariationOperator):
    """Chains together an arbitrary number of VariationOperators to produce a single
    child.

    Parameters
    ----------
    operators : tuple or list
        Tuple or list of variation operators to chain together in order.

    """

    def __init__(self, operators):
        self.operators = operators

    def produce(self, individual1, individual2, function_set):
        """
        Produces a child using all of the operators in the pipeline.

        Parameters
        ----------
        individual1 : Individual
            The first Individual whose program will be used during alternation
            to produce a child program.

        individual2 : Individual
            The second Individual whose program will be used during alternation
            to produce a child program.

        function_set : list[]
            List of function names (strings) to use when mutation overwrites
            code in an individual's program with new code. This should included
            supported function names and input_n where n is the index of a
            feature.
        """
        child = individual1
        for op in self.operators:
            if op.operator_type == 'mutation':
                child = op.produce(child, function_set)
            elif op.operator_type == 'recombination':
                child = op.produce(child, individual2)
            elif op.operator_type == 'pipeline':
                child = op.produce(child, individual2, function_set)
        return child
