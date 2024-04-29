import os
import json

import matplotlib.pyplot as plt
import pandas as pd
from pgmpy.factors.discrete import (TabularCPD)
from pgmpy.models import BayesianNetwork

from src import PROJECT_PATH
from src.create_synth_pop import CondCreat
from src.dataloader import Downloader
from src.gibbs_sampling import NewGibbsSampling
from src.validation import Validation

import warnings
warnings.filterwarnings("ignore")


class SyntheticPopulationGenerator:
    def __init__(self, full_names, partial_1):
        Downloader(gdrive_id="1fni7wudNdWjy5BsPpNXaxLGshyNtiapK", file_name="Census_2016_Individual_PUMF.dta")
        Downloader(gdrive_id="1JCPZtvA3oxdBFRjB_MTB3AWVz02zz3Z8", file_name="ipf_10.json")
        Downloader(gdrive_id="1vbMwVXsVg8mFDUxq-ioPpVZPt6Wrmqwv", file_name="da_10.json")
        Downloader(gdrive_id="1B0vWMONt6MFSzld_CghvzQMN_iyH1I7R", file_name="ipf_10pr.csv")
        Downloader(gdrive_id="1IKBeruFOMWjriZWYGvZs3mf77-M5bnLl", file_name="evidence_partial_1.json")
        Downloader(gdrive_id="1UTLwhVQdAmx8xj9iufmOkEYAUi1CoySB", file_name="evidence.json")

        self.table = Downloader.read_data(file=os.path.join(PROJECT_PATH, "data/Census_2016_Individual_PUMF.dta"))
        with open(os.path.join(PROJECT_PATH, "data/evidence.json")) as json_file:
            self.full_evidence = json.load(json_file)
        with open(os.path.join(PROJECT_PATH, "data/evidence_partial_1.json")) as json_fil:
            self.partial_evidence = json.load(json_fil)
        self.full_names = full_names
        self.partial_1 = partial_1

    def run_cond_creat(self):
        cla = CondCreat(table=self.table, full_evidence=self.full_evidence,
                        partial_evidence=self.partial_evidence, full_names=self.full_names,
                        save=True, partial_1=self.partial_1)

        return cla

    def run_gibbs_sampling_census(self):
        data_generated = os.path.join(PROJECT_PATH, "generated")
        cond_1 = pd.read_csv(os.path.join(data_generated, "full_cross_table_1.csv"), header=[0, 1, 2, 3, 4],
                             index_col=[0])
        cond_2 = pd.read_csv(os.path.join(data_generated, "full_cross_table_2.csv"), header=[0, 1, 2, 3, 4],
                             index_col=[0])
        cond_3 = pd.read_csv(os.path.join(data_generated, "full_cross_table_3.csv"), header=[0, 1, 2, 3, 4],
                             index_col=[0])
        cond_4 = pd.read_csv(os.path.join(data_generated, "full_cross_table_4.csv"), header=[0, 1, 2, 3, 4],
                             index_col=[0])
        cond_5 = pd.read_csv(os.path.join(data_generated, "full_cross_table_5.csv"), header=[0, 1, 2, 3, 4],
                             index_col=[0])
        cond_6 = pd.read_csv(os.path.join(data_generated, "full_cross_table_6.csv"), header=[0, 1, 2, 3, 4],
                             index_col=[0])
        partial = pd.read_csv(os.path.join(data_generated, "partial_cross_1.csv"), header=[0, 1, 2, 3],
                              index_col=[0])

        student = BayesianNetwork([("agegrp", "Sex"), ("hdgree", "lfact"), ("TotInc", "hhsize")])
        cpd_age = TabularCPD(variable="agegrp", variable_card=18, values=cond_1.values,
                             evidence=self.full_evidence["agegrp"], evidence_card=[2, 3, 3, 4, 5])
        cpd_sex = TabularCPD(variable="Sex", variable_card=2, values=cond_2.values,
                             evidence=self.full_evidence["Sex"], evidence_card=[18, 3, 3, 4, 5])
        cpd_hdgree = TabularCPD(variable="hdgree", variable_card=3, values=cond_3.values,
                                evidence=self.full_evidence["hdgree"], evidence_card=[18, 2, 3, 4, 5])
        cpd_lfact = TabularCPD(variable="lfact", variable_card=3, values=cond_4.values,
                               evidence=self.full_evidence["lfact"], evidence_card=[18, 2, 3, 4, 5])
        cpd_totinc = TabularCPD(variable="TotInc", variable_card=4, values=cond_5.values,
                                evidence=self.full_evidence["TotInc"], evidence_card=[18, 2, 3, 3, 5])
        cpd_hhsize = TabularCPD(variable="hhsize", variable_card=5, values=cond_6.values,
                                evidence=self.full_evidence["hhsize"], evidence_card=[18, 2, 3, 3, 4])
        student.add_cpds(cpd_age, cpd_sex, cpd_lfact, cpd_totinc, cpd_hdgree, cpd_hhsize)

        gibbs_chain = NewGibbsSampling(evidence=self.full_evidence, model=student)
        samples = gibbs_chain.sample(size=9165670)
        samples.drop(samples.index[0:50000], axis=0, inplace=True)
        samples = samples.iloc[::20, :]
        samples.to_csv(os.path.join(data_generated, "samples.csv"), index=False)

        # it is needed to give which feature is independent of the studied one,
        # what its cardinality is, and where is its position in the evidence dictionary[name]
        partial_1 = {"agegrp": ["Sex", 2, 0]}

        population_partial = BayesianNetwork([("agegrp", "Sex"), ("hdgree", "lfact"), ("TotInc", "hhsize")])
        cpd_partial_1 = TabularCPD(variable="agegrp", variable_card=18, values=partial.values,
                                   evidence=self.partial_evidence["agegrp"], evidence_card=[3, 3, 4, 5])
        population_partial.add_cpds(cpd_partial_1, cpd_sex, cpd_lfact, cpd_totinc, cpd_hdgree, cpd_age)
        gibbs_chain_partial = NewGibbsSampling(evidence=self.partial_evidence,
                                               model=population_partial, partial=partial_1)
        samples_partial = gibbs_chain_partial.sample(size=9165670)
        print(type(samples_partial))
        samples_partial = samples_partial.iloc[::10, :]
        print(samples_partial.shape)
        samples_partial.to_csv(os.path.join(data_generated, "samples_partial.csv"), index=False)

    def run_validation(self):
        gender_1 = {"Sex": ["female", "male"]}
        gender_2 = {"Sex": ["female", "male"]}

        age_group_1 = {"agegrp": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "40-44", "45-49", "50-59",
                                  "60-64", "65-69", "70-74", "75-79" "80-84", "85-89", "90-94", "95-99", "100"]}
        age_group_2 = {"agegrp": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "40-44", "45-49", "50-59",
                                  "60-64", "65-69", "70-74", "75-79" "80-84", "85-89", "90-94", "95-99", "100"]}

        hdgree_1 = {"hdgree": ["no", "secondary", "university"]}
        hdgree_2 = {"hdgree": ["no", "secondary", "university"]}

        hhsize_1 = {"hhsize": ["1", "2", "3", "4", "5+"]}
        hhsize_2 = {"hhsize": ["1", "2", "3", "4", "5+"]}

        validation = Validation(
            creat_class=CondCreat(table=self.table, full_evidence=self.full_evidence,
                                  partial_evidence=self.partial_evidence, full_names=self.full_names,
                                  save=False, partial_1=self.partial_1),
            names=self.full_names)
        validation.plot_figures(col_name="Sex", dict_1=gender_1, dict_2=gender_2)
        validation.plot_figures(col_name="agegrp", dict_1=age_group_1, dict_2=age_group_2)
        validation.plot_figures(col_name="hdgree", dict_1=hdgree_1, dict_2=hdgree_2)
        validation.plot_figures(col_name="hhsize", dict_1=hhsize_1, dict_2=hhsize_2)

        validation.plot_lin_regression(x=validation.cross_gibbs, y=validation.cross_census,
                                       xlabel="Simulation", title="Full conditionals")

        validation.plot_lin_regression(x=validation.cross_partial_1, y=validation.cross_census,
                                       xlabel="Simulation", title="Partial_1")

        plt.show()

        r, nrmse, rae = validation.run_calculations(census=validation.cross_census,
                                                    simulation=validation.cross_gibbs)
        print(r, nrmse, rae)


def main():
    names = ["agegrp", "Sex", "hdgree", "lfact", "TotInc", "hhsize"]
    partial_1 = ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"]

    generate_synth = SyntheticPopulationGenerator(full_names=names, partial_1=partial_1)

    generate_synth.run_cond_creat()
    generate_synth.run_validation()


if __name__ == "__main__":
    main()
