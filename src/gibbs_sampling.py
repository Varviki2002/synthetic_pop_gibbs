from collections import namedtuple
import itertools
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import _return_samples
from pgmpy.sampling.Sampling import GibbsSampling
from pgmpy.utils.mathext import sample_discrete
from tqdm.auto import tqdm

import warnings

warnings.filterwarnings("ignore")

State = namedtuple(typename="State", field_names=["var", "state"])


class NewGibbsSampling(GibbsSampling):
    def __init__(self, evidence, model, partial=None):
        super().__init__(model=model)

        self.evidence = evidence
        self.partial = partial

        if isinstance(model, BayesianNetwork):
            self._get_from_bayesian_model(model=model)

    def _get_from_bayesian_model(self, model):
        """
        Computes the Gibbs transition models from a Bayesian Network.
        "Probabilistic Graphical Model Principles and Techniques", Koller and
        Friedman, Section 12.3.3 pp 512-513.

        Parameters
        ----------
        model: BayesianNetwork
            The model from which probabilities will be computed.
        """
        self.variables = np.array(model.nodes())
        self.latents = model.latents
        self.cardinalities = {
            var: model.get_cpds(var).variable_card for var in self.variables
        }

        for var in self.variables:
            other_vars = self.evidence[var]
            other_cards = [self.cardinalities[v] for v in other_vars]
            kernel = {}
            lista = self.create_lists(other_cards=other_cards)
            for tup in itertools.product(*lista):
                array = np.zeros(self.cardinalities[var])
                if (self.partial is not None) and (var in list(self.partial.keys())):
                    for i in range(self.cardinalities[var]):
                        cpd = model.get_cpds(node=var).values
                        array[i] = cpd[i][tup[0]][tup[1]][tup[2]][tup[3]]
                    for num in range(self.partial[var][1]):
                        list_tup = list(tup)
                        list_tup.insert(self.partial[var][2], num)
                        tup_kernel = tuple(list_tup)
                        kernel[tup_kernel] = array
                else:
                    for i in range(self.cardinalities[var]):
                        cpd = model.get_cpds(node=var).values
                        array[i] = cpd[i][tup[0]][tup[1]][tup[2]][tup[3]][tup[4]]
                    kernel[tup] = array
            self.transition_models[var] = kernel

    @staticmethod
    def create_lists(other_cards):
        lista = []
        for item in other_cards:
            small_list = []
            for i in range(item):
                small_list.append(i)
            lista.append(small_list)
        return lista

    def sample(self, start_state=None, size=1, seed=None, include_latents=False):
        if start_state is None and self.state is None:
            self.state = self.random_state()
        elif start_state is not None:
            self.set_start_state(start_state)

        if seed is not None:
            np.random.seed(seed)

        types = [(str(var_name), "int") for var_name in self.variables]
        sampled = np.zeros(size, dtype=types).view(np.recarray)
        sampled[0] = tuple(st for var, st in self.state)
        for i in tqdm(range(size - 1)):
            for j, (var, st) in enumerate(self.state):
                other_st = tuple(st for v, st in self.state if var != v)
                next_st = sample_discrete(
                    list(range(self.cardinalities[var])),
                    self.transition_models[var][other_st],
                )[0]
                self.state[j] = State(var, next_st)
            sampled[i + 1] = tuple(st for var, st in self.state)

        samples_df = _return_samples(sampled)
        if not include_latents:
            samples_df.drop(self.latents, axis=1, inplace=True)
        return samples_df

