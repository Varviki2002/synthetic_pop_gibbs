import itertools
import numpy as np
import pandas as pd
from pgmpy.factors.discrete import (TabularCPD)
from pgmpy.models import BayesianNetwork
from pgmpy.sampling.Sampling import GibbsSampling

import warnings
warnings.filterwarnings('ignore')


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
                    array[i] = cpd[i][tup[0]][tup[1]][tup[2]]
                kernel[tup] = array
            self.transition_models[var] = kernel


def main():
    evidence = {"agegrp": ["sex", "hhsize", "hdgree"], "sex": ["agegrp", "hhsize", "hdgree"],
                "hdgree": ["agegrp", "sex", "hhsize"], "hhsize": ["agegrp", "sex", "hdgree"]}

    cond_1 = pd.read_excel("../generated/cross_table_1.xlsx", header=[0, 1, 2], index_col=[0])
    cond_2 = pd.read_excel("../generated/cross_table_2.xlsx", header=[0, 1, 2], index_col=[0])
    cond_3 = pd.read_excel("../generated/cross_table_3.xlsx", header=[0, 1, 2], index_col=[0])
    cond_4 = pd.read_excel("../generated/cross_table_4.xlsx", header=[0, 1, 2], index_col=[0])

    student = BayesianNetwork([('agegrp', 'sex'), ('hhsize', 'hdgree')])
    cpd_diff = TabularCPD('agegrp', 18, cond_1.values, evidence["agegrp"], [2, 5, 3])
    cpd_intel = TabularCPD('sex', 2, cond_2.values, evidence["sex"], [18, 5, 3])
    cpd_grade = TabularCPD('hdgree', 3, cond_3.values, evidence["hdgree"], [18, 2, 5])
    cpd_age = TabularCPD('hhsize', 5, cond_4.values, evidence["hhsize"], [18, 2, 3])
    student.add_cpds(cpd_diff, cpd_intel, cpd_grade, cpd_age)

    gibbs_chain = NewGibbsSampling(evidence=evidence, model=student)
    samples = gibbs_chain.sample(size=14826299)
    print(type(samples))
    samples.to_csv("../data/samples.csv", index=False)


if __name__ == "__main__":
    main()
