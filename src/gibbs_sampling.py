import os

from pgmpy.factors.discrete import (TabularCPD)
import numpy as np
import itertools
from pgmpy.sampling.Sampling import State, GibbsSampling

import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.factors import factor_product
from pgmpy.models import BayesianNetwork, MarkovChain, MarkovNetwork
from pgmpy.sampling import BayesianModelInference, _return_samples
from pgmpy.utils.mathext import sample_discrete, sample_discrete_maps

from src.dataloader import Downloader
from src.create_synth_pop import CondCreat

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
        evidence = self.evidence
        self.latents = model.latents
        self.cardinalities = {
            var:  model.get_cpds(var).variable_card for var in self.variables
        }
        dimension = len(self.variables)

        for var in self.variables:
            other_vars = self.evidence[var]
            other_cards = [self.cardinalities[v] for v in other_vars]
            kernel = {}
            # factors = [cpd.to_factor() for cpd in model.cpds if var in cpd.scope()]
            # factor = factor_product(*factors)
            # scope = set(factor.scope())
            lista = []
            for item in other_cards:
                small_list = []
                for i in range(item):
                    small_list.append(i)
                lista.append(small_list)
            # scope = {v for v in self.variables if v is not None}
            for tup in itertools.product(*lista):
                # states = [State(v, s) for v, s in zip(other_vars, tup) if v in scope]
                array = np.zeros(self.cardinalities[var])
                for i in range(self.cardinalities[var]):
                    cpd = model.get_cpds(node=var).values
                    array[i] = cpd[i][tup[0]][tup[1]][tup[2]]
                # reduced_factor = factor.reduce(states, inplace=False)
                kernel[tup] = array
            self.transition_models[var] = kernel



def main():
    path = os.getcwd()
    data = pd.read_csv("../data/synthetic_pop_y_2021.csv")

    evidence = {"agegrp": ["sex", "hhsize", "hdgree"], "sex": ["agegrp", "hhsize", "hdgree"],
                "hdgree": ["agegrp", "sex", "hhsize"], "hhsize": ["agegrp", "sex", "hdgree"]}

    cla = CondCreat(table=data, evidence=evidence)

    cond_1, cond_2, cond_3, cond_4 = cla.create_conditional_tables()




    student = BayesianNetwork([('agegrp', 'sex'), ('hhsize', 'hdgree')])
    cpd_diff = TabularCPD('agegrp', 18, cond_1.values, evidence["agegrp"], [2, 5, 3])
    cpd_intel = TabularCPD('sex', 2, cond_2.values, evidence["sex"], [18, 5, 3])
    cpd_grade = TabularCPD('hdgree', 3, cond_3.values, evidence["hdgree"], [18, 2, 5])
    cpd_age = TabularCPD('hhsize', 5, cond_4.values, evidence["hhsize"], [18, 2, 3])
    student.add_cpds(cpd_diff, cpd_intel, cpd_grade, cpd_age)
    # student.get_cpds('agegrp', evidence=evidence["agegrp"])
    # student.get_cpds('sex', evidence=evidence["sex"])
    # student.get_cpds('hdgree', evidence=evidence["hdgree"])
    # student.get_cpds('hhsize', evidence=evidence["hhsize"])

    gibbs_chain = NewGibbsSampling(evidence=evidence, model=student)
    samples = gibbs_chain.sample(size=14826299)
    print(type(samples))
    # samples = samples.drop(samples[0:20001], axis=0, inplace=True)
    samples.to_csv("data/samples.csv")


if __name__ == "__main__":
    main()
