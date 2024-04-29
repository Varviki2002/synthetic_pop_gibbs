from collections import namedtuple
import itertools
import os
import warnings

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import (TabularCPD)
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import _return_samples
from pgmpy.sampling.Sampling import GibbsSampling
from pgmpy.utils.mathext import sample_discrete
from tqdm.auto import tqdm

from src import PROJECT_PATH

warnings.filterwarnings("ignore")
State = namedtuple(typename="State", field_names=["var", "state"])


class NewGibbsSampling(GibbsSampling):
    def __init__(self, evidence, model, partial=None):
        super().__init__()

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


def main():
    evidence = {"agegrp": ["Sex", "hdgree", "lfact", "TotInc", "hhsize"],
                "Sex": ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"],
                "hdgree": ["agegrp", "Sex", "lfact", "TotInc", "hhsize"],
                "lfact": ["agegrp", "Sex", "hdgree", "TotInc", "hhsize"],
                "TotInc": ["agegrp", "Sex", "hdgree", "lfact", "hhsize"],
                "hhsize": ["agegrp", "Sex", "hdgree", "lfact", "TotInc"]}

    # it is needed to give which feature is independent of the studied one,
    # what its cardinality is, and where is its position in the evidence dictionary[name]
    partial_1 = {"agegrp": ["Sex", 2, 0]}

    evidence_partial_1 = {"agegrp": ["hdgree", "lfact", "TotInc", "hhsize"],
                          "Sex": ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"],
                          "hdgree": ["agegrp", "Sex", "lfact", "TotInc", "hhsize"],
                          "lfact": ["agegrp", "Sex", "hdgree", "TotInc", "hhsize"],
                          "TotInc": ["agegrp", "Sex", "hdgree", "lfact", "hhsize"],
                          "hhsize": ["agegrp", "Sex", "hdgree", "lfact", "TotInc"]}

    data_generated = os.path.join(PROJECT_PATH, "generated")

    cond_1 = pd.read_csv(os.path.join(data_generated, "full_cross_table_1.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_2 = pd.read_csv(os.path.join(data_generated, "full_cross_table_2.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_3 = pd.read_csv(os.path.join(data_generated, "full_cross_table_3.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_4 = pd.read_csv(os.path.join(data_generated, "full_cross_table_4.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_5 = pd.read_csv(os.path.join(data_generated, "full_cross_table_5.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_6 = pd.read_csv(os.path.join(data_generated, "full_cross_table_6.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    partial = pd.read_csv(os.path.join(data_generated, "partial_cross_1.csv"), header=[0, 1, 2, 3], index_col=[0])

    student = BayesianNetwork([("agegrp", "Sex"), ("hdgree", "lfact"), ("TotInc", "hhsize")])
    cpd_age = TabularCPD(variable="agegrp", variable_card=18, values=cond_1.values, evidence=evidence["agegrp"],
                         evidence_card=[2, 3, 3, 4, 5])
    cpd_sex = TabularCPD(variable="Sex", variable_card=2, values=cond_2.values, evidence=evidence["Sex"],
                         evidence_card=[18, 3, 3, 4, 5])
    cpd_hdgree = TabularCPD(variable="hdgree", variable_card=3, values=cond_3.values, evidence=evidence["hdgree"],
                            evidence_card=[18, 2, 3, 4, 5])
    cpd_lfact = TabularCPD(variable="lfact", variable_card=3, values=cond_4.values, evidence=evidence["lfact"],
                           evidence_card=[18, 2, 3, 4, 5])
    cpd_totinc = TabularCPD(variable="TotInc", variable_card=4, values=cond_5.values, evidence=evidence["TotInc"],
                            evidence_card=[18, 2, 3, 3, 5])
    cpd_hhsize = TabularCPD(variable="hhsize", variable_card=5, values=cond_6.values, evidence=evidence["hhsize"],
                            evidence_card=[18, 2, 3, 3, 4])
    student.add_cpds(cpd_age, cpd_sex, cpd_lfact, cpd_totinc, cpd_hdgree, cpd_hhsize)

    # gibbs_chain = NewGibbsSampling(evidence=evidence, model=student)
    # samples = gibbs_chain.sample(size=9215670)
    # print(type(samples))
    # samples.drop(samples.index[0:50000], axis=0, inplace=True)
    # samples = samples.iloc[::10, :]
    # print(samples.shape)
    # samples.to_csv(os.path.join(data_generated, "samples.csv"), index=False)

    population_partial = BayesianNetwork([("agegrp", "Sex"), ("hdgree", "lfact"), ("TotInc", "hhsize")])
    cpd_partial_1 = TabularCPD(variable="agegrp", variable_card=18, values=partial.values,
                               evidence=evidence_partial_1["agegrp"], evidence_card=[3, 3, 4, 5])
    population_partial.add_cpds(cpd_partial_1, cpd_sex, cpd_lfact, cpd_totinc, cpd_hdgree, cpd_hhsize)
    gibbs_chain_partial = NewGibbsSampling(evidence=evidence_partial_1, model=population_partial, partial=partial_1)
    samples_partial = gibbs_chain_partial.sample(size=9165670)
    print(type(samples_partial))
    samples_partial = samples_partial.iloc[::10, :]
    print(samples_partial.shape)
    samples_partial.to_csv(os.path.join(data_generated, "samples_partial.csv"), index=False)


if __name__ == "__main__":
    main()
