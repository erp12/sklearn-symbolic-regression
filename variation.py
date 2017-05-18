"""
TODO
"""
from abc import ABCMeta, abstractmethod
import random

from .population import FUNCS, ERC_GENERATORS, generate_random_code, Individual, Population
from .utils import get_arity, noise_factor

class VariationOperator(metaclass=ABCMeta):
    """TODO: Write docstring
    """

    _operator_type = None

    @property
    def operator_type(self):
        return self._operator_type

    @operator_type.setter
    def operator_type(self, value):
        if value not in ('mutation', 'recombination'):
            raise AttributeError('Unknown variation operator type' + str(value))
        self._operator_type = value

    @operator_type.deleter
    def operator_type(self):
        del self._operator_type

    @abstractmethod
    def produce(self, individual_1, individual_2):
        """Produces a child.

        :param list individual1: Individual of parent 1.
        :param list individual2: Individual of parent 2.
        :returns: A new Individual created by alternating between parent programs.
        """

class UniformMutator(VariationOperator):
    """TODO: Write docstring
    """

    def __init__(self, rate=0.1, constant_perturb_rate=0.5,
        perturb_standard_deviation=1.0):
        self.rate = rate
        self.constant_perturb_rate = constant_perturb_rate
        self.perturb_standard_deviation = perturb_standard_deviation
        self.operator_type = 'mutation'

    def produce(self, individual_1, individual_2=None, function_set=None):
        """TODO: Write docstring
        """
        new_prog = []
        for el in individual_1.program[:]:
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

    More information can be found on the `this Push-Redux page
    <https://erp12.github.io/push-redux/pages/genetic_operators/index.html#recombination>`_.
    """

    def __init__(self, rate=0.1, alignment_deviation=10):
        self.rate = rate
        self.alignment_deviation = alignment_deviation
        self.operator_type = 'recombination'

    def produce(self, individual1, individual2):
        """Produces a child using the alternation operator.

        :param list individual1: Individual of parent 1.
        :param list individual2: Individual of parent 2.
        :returns: A new Individual created by alternating between parent programs.
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

    pass
