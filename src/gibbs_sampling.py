import os
from collections import namedtuple

import itertools
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import (TabularCPD)
from pgmpy.models import BayesianNetwork
from pgmpy.sampling.Sampling import GibbsSampling
from pgmpy.utils.mathext import sample_discrete
from tqdm.auto import tqdm
from pgmpy.sampling import _return_samples

from src import PROJECT_PATH
from src.dataloader import Downloader
from src.create_synth_pop import CondCreat

import warnings

warnings.filterwarnings('ignore')

State = namedtuple("State", ["var", "state"])


class NewGibbsSampling(GibbsSampling):
    def __init__(self, evidence, model):
        super().__init__(model)

        self.evidence = evidence

        if isinstance(model, BayesianNetwork):
            self._get_from_bayesian_model(model)

    def _get_from_bayesian_model(self, model):
        """
        Computes the Gibbs transition models from a Bayesian Network.
        'Probabilistic Graphical Model Principles and Techniques', Koller and
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
            lista = []
            for item in other_cards:
                small_list = []
                for i in range(item):
                    small_list.append(i)
                lista.append(small_list)
            for tup in itertools.product(*lista):
                array = np.zeros(self.cardinalities[var])
                for i in range(self.cardinalities[var]):
                    cpd = model.get_cpds(node=var).values
                    array[i] = cpd[i][tup[0]][tup[1]][tup[2]][tup[3]][tup[4]]
                kernel[tup] = array
            self.transition_models[var] = kernel

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
    data = Downloader.read_data(file=os.path.join(PROJECT_PATH, "data/Census_2016_Individual_PUMF.dta"))

    names = ["agegrp", "Sex", "hdgree", "lfact", "TotInc", "hhsize"]
    cards = [18, 2, 3, 3, 4, 5]

    evidence = {"agegrp": ["Sex", "hdgree", "lfact", "TotInc", "hhsize"],
                "Sex": ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"],
                "hdgree": ["agegrp", "Sex", "lfact", "TotInc", "hhsize"],
                "lfact": ["agegrp", "Sex", "hdgree", "TotInc", "hhsize"],
                "TotInc": ["agegrp", "Sex", "hdgree", "lfact", "hhsize"],
                "hhsize": ["agegrp", "Sex", "hdgree", "lfact", "TotInc"]}

    # cla = CondCreat(table=data, evidence=evidence, full_names=names)

    data_generated = os.path.join(PROJECT_PATH, "generated")

    # cond_1, cond_2, cond_3, cond_4, cond_5, cond_6 = cla.create_full_conditional_tables()
    # cond_1.fillna(0.0, inplace=True)
    # cond_2.fillna(0.0, inplace=True)
    # cond_3.fillna(0.0, inplace=True)
    # cond_4.fillna(0.0, inplace=True)
    # cond_5.fillna(0.0, inplace=True)
    # cond_6.fillna(0.0, inplace=True)


    cond_1 = pd.read_csv(os.path.join(data_generated, "full_cross_table_1.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_2 = pd.read_csv(os.path.join(data_generated, "full_cross_table_2.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_3 = pd.read_csv(os.path.join(data_generated, "full_cross_table_3.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_4 = pd.read_csv(os.path.join(data_generated, "full_cross_table_4.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_5 = pd.read_csv(os.path.join(data_generated, "full_cross_table_5.csv"), header=[0, 1, 2, 3, 4], index_col=[0])
    cond_6 = pd.read_csv(os.path.join(data_generated, "full_cross_table_6.csv"), header=[0, 1, 2, 3, 4], index_col=[0])

    student = BayesianNetwork([("agegrp", "Sex"), ("hdgree", "lfact"), ("TotInc", "hhsize")])
    cpd_diff = TabularCPD('agegrp', 18, cond_1.values, evidence["agegrp"],
                          [2, 3, 3, 4, 5])
    cpd_intel = TabularCPD('Sex', 2, cond_2.values, evidence["Sex"],
                           [18, 3, 3, 4, 5])
    cpd_grade = TabularCPD('hdgree', 3, cond_3.values, evidence["hdgree"],
                           [18, 2, 3, 4, 5])
    cpd_lfact = TabularCPD('lfact', 3, cond_4.values, evidence["lfact"],
                           [18, 2, 3, 4, 5])
    cpd_totinc = TabularCPD('TotInc', 4, cond_5.values, evidence["TotInc"],
                            [18, 2, 3, 3, 5])
    cpd_age = TabularCPD('hhsize', 5, cond_6.values, evidence["hhsize"],
                         [18, 2, 3, 3, 4])
    student.add_cpds(cpd_diff, cpd_intel, cpd_lfact, cpd_totinc, cpd_grade, cpd_age)

    gibbs_chain = NewGibbsSampling(evidence=evidence, model=student)
    samples = gibbs_chain.sample(size=9165670)
    print(type(samples))
    samples = samples.iloc[::10, :]
    print(samples.shape)
    samples.to_csv(os.path.join(data_generated, "samples.csv"), index=False)


if __name__ == "__main__":
    main()
