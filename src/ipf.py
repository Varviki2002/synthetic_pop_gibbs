import os

import humanleague
import json
import numpy as np
import pandas as pd

from src import PROJECT_PATH
from src.dataloader import Downloader


class IPF:
    def __init__(self, table, json_dict, province):
        self.province = province
        self.json_dict = json_dict
        self.table = table
        self.seed = self._load_seed(dataframe=self.table)

    @staticmethod
    def unlistify(table, columns, sizes, values):
        """
        Converts an n-column table of counts into an n-dimensional array of counts
        """
        pivot = table.pivot_table(index=columns, values=values, aggfunc="sum")
        # order must be same as column order above
        array = np.zeros(sizes, dtype=int)
        array[tuple(pivot.index.codes)] = pivot.values.flat
        return array

    # Load seed from microsample
    def _load_seed(self, dataframe):
        n_sex = len(dataframe["Sex"].unique())
        n_age = len(dataframe["agegrp"].unique())
        n_hdgree = len(dataframe["hdgree"].unique())
        n_lfact = len(dataframe["lfact"].unique())
        n_hhsize = len(dataframe["hhsize"].unique())
        n_totinc = len(dataframe["TotInc"].unique())

        cols = ["Sex", "agegrp", "hdgree", "lfact", "TotInc", "hhsize"]
        shape = [n_sex, n_age, n_hdgree, n_lfact, n_totinc, n_hhsize]

        seed = self.unlistify(dataframe, cols, shape, "weight")
        seed = seed.astype(float) + 1.0
        seed = seed * self.get_impossible(seed)

        # Convergence problems can occur when one of the rows is zero yet the marginal total is nonzero.
        # Can get round this by adding a small number to the seed effectively allowing zero states
        #  to be occupied with a finite probability
        return seed

    @staticmethod
    def get_impossible(seed):
        # zeros out impossible states, all others are equally probable
        constraints = np.ones(seed.shape)
        # Add impossible constraints:
        # hdgree >0 and age 0 to 2,
        constraints[:, 0, 1, :, :, :] = 0
        constraints[:, 0, 2, :, :, :] = 0
        constraints[:, 1, 1, :, :, :] = 0
        constraints[:, 1, 2, :, :, :] = 0
        constraints[:, 2, 1, :, :, :] = 0
        constraints[:, 2, 2, :, :, :] = 0
        # employed or unemployed and age 0 to 2
        constraints[:, 0, :, 1, :, :] = 0
        constraints[:, 0, :, 2, :, :] = 0
        constraints[:, 1, :, 1, :, :] = 0
        constraints[:, 1, :, 2, :, :] = 0
        constraints[:, 2, :, 1, :, :] = 0
        constraints[:, 2, :, 2, :, :] = 0
        # hhsize 0 (1p) and age 0 to 2
        constraints[:, 0, :, :, :, 2] = 0
        constraints[:, 1, :, :, :, 1] = 0
        constraints[:, 2, :, :, :, 2] = 0
        # totinc >0 and age 0 to 2
        for i in range(1, 4):
            constraints[:, 0, :, :, i, :] = 0
            constraints[:, 1, :, :, i, :] = 0
            constraints[:, 2, :, :, i, :] = 0

        return constraints

    def gather_marginals(self):
        total_male = self.json_dict["total_male"]
        total_female = self.json_dict["total_female"]
        total_age = self.json_dict["total_age"]
        total_age_f = self.json_dict["total_age_f"]
        total_age_m = self.json_dict["total_age_m"]
        total_hdgree = self.json_dict["total_hdgree"]
        total_hh_size = self.json_dict["total_hh_size"]
        total_lfact = self.json_dict["total_lfact"]
        total_inc = self.json_dict["total_inc"]

        print("Gather marginals...")
        # get marginal by sex, by prihm, by age, by agebysex, hdgree
        # 0:F 1:M
        marginal_sex = np.array([total_female, total_male])
        # 0:0-4y ... 17: 85+
        marginal_age = np.array(list(total_age.values()))
        # 0: F age, 1: M age
        marginal_age_by_sex = np.array([list(total_age_f.values()),
                                        list(total_age_m.values())])
        # 0: no, 1:secondary, 2: university
        marginal_hdgree = np.array(list(total_hdgree.values()))
        # 0: employed, 1:unemployed, 2: not in labour force
        marginal_lfact = np.array(list(total_lfact.values()))
        # 0: 1; 1: 2; 2: 3; 3: 4; 4: 5+
        marginal_hh_size = np.array(list(total_hh_size.values()))
        # <20k, 20-60k, 60-100k, 100+
        marginal_inc = np.array(list(total_inc.values()))
        return (marginal_sex, marginal_age, marginal_age_by_sex, marginal_hdgree,
                marginal_hh_size, marginal_lfact, marginal_inc)

    @staticmethod
    def probabilistic_sampling(p, total_pop):
        probas = np.float64(p[0]).ravel()
        probas /= np.sum(probas)
        selected = np.random.choice(len(probas), total_pop, True, probas)
        result = np.zeros(p[0].shape, np.uint8)
        result.ravel()[selected] = 1
        return result

    def ipf(self):
        total_pop = self.json_dict["total_pop"]

        i0 = np.array([0])
        i1 = np.array([1])
        i2 = np.array([0, 1])
        i3 = np.array([3])
        i4 = np.array([4])
        i5 = np.array([5])
        i6 = np.array([6])

        (marginal_sex, marginal_age, marginal_age_by_sex, marginal_hdgree,
         marginal_hh_size, marginal_lfact, marginal_inc) = self.gather_marginals()

        print("Apply IPF (could be replaced by qisi for more accurate.)")
        p = humanleague.ipf(
            self.seed, indices=[i0, i1, i2, i3, i4, i5, i6],
            marginals=[marginal_sex.astype(float), marginal_age.astype(float),
                       marginal_age_by_sex.astype(float), marginal_hdgree.astype(float),
                       marginal_lfact.astype(float), marginal_inc.astype(float),
                       marginal_hh_size.astype(float)])
        # p = np.load(os.path.join(PROJECT_PATH, "generated/p.npy"))
        p_list = list(p)
        p_list[0] = self.probabilistic_sampling(p, total_pop)

        p = tuple(p_list)
        return p

    def create_synth_pop(self):
        p = self.ipf()

        chunk = pd.DataFrame(
            columns=["Sex", "agegrp", "hdgree", "lfact", "TotInc", "hhsize", "province"])

        syn_pop = pd.DataFrame(columns=["Sex", "agegrp", "hdgree", "lfact", "TotInc", "hhsize", "province"])

        table = humanleague.flatten(p[0])
        # table = np.load(os.path.join(PROJECT_PATH, "generated/table.npy"))
        chunk.Sex = table[0]
        chunk.agegrp = table[1]
        chunk.hdgree = table[2]
        chunk.lfact = table[3]
        chunk.hhsize = table[5]
        chunk.TotInc = table[4]
        chunk["province"] = self.province
        syn_pop = pd.concat([syn_pop, chunk], ignore_index=True)
        return syn_pop


def main():
    f = open(os.path.join(PROJECT_PATH, "data/ipf_10.json"))
    json_dict = json.load(f)

    data = Downloader.read_data(file=os.path.join(PROJECT_PATH, "data/Census_2016_Individual_PUMF.dta"))

    # names = ["agegrp", "Sex", "hdgree", "lfact", "TotInc", "hhsize"]
    # ipf = IPF(table=data, json_dict=json_dict, province=10)
    # ipf_data = ipf.create_synth_pop()
    print(data)


if __name__ == "__main__":
    main()
