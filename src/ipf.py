import os

import humanleague
import numpy as np
import pandas as pd

from src import PROJECT_PATH


class IPF:
    def __init__(self, table, name_list, sizes):
        self.table = table
        self.name_list = name_list
        self.sizes = sizes
        self.seed = np.zeros(self.sizes, dtype=int)
        self.p = self.ipf()

    def create_seed(self):
        """
        Converts an n-column table of counts into an n-dimensional array of counts
        """

        seed = np.zeros(self.sizes, dtype=int)
        seed = seed.astype(float) + 1.0  # / np.sum(seed)
        seed = seed * self.get_impossible(seed)

        return seed

    def get_impossible(self, seed):
        # zeros out impossible states, all others are equally probable
        constraints = np.ones(seed.shape)
        # hdgree >0 and age 0 to 2,
        constraints[:, 0, 1, :] = 0
        constraints[:, 0, 2, :] = 0
        constraints[:, 1, 1, :] = 0
        constraints[:, 1, 2, :] = 0
        constraints[:, 2, 1, :] = 0
        constraints[:, 2, 2, :] = 0
        # hhsize 0 (1p) and age 0 to 2
        constraints[:, 0, :, 0] = 0
        constraints[:, 1, :, 0] = 0
        constraints[:, 2, :, 0] = 0

        return constraints

    def create_marginals(self):
        total_female = len(self.table[self.table["sex"] == 0])
        total_male = len(self.table[self.table["sex"] == 1])
        total_age = {}
        for i in range(0, self.sizes[1]):
            l = len(self.table[self.table["agegrp"] == i])
            total_age[i] = l

        total_age_f = {}
        for i in range(0, self.sizes[1]):
            l = len(self.table.loc[(self.table['agegrp'] == i) & (self.table['sex'] == 0)])
            total_age_f[i] = l

        total_age_m = {}
        for i in range(0, self.sizes[1]):
            l = len(self.table.loc[(self.table['agegrp'] == i) & (self.table['sex'] == 1)])
            total_age_m[i] = l

        total_hdgree = {}
        for i in range(0, self.sizes[2]):
            l = len(self.table.loc[self.table['hdgree'] == i])
            total_hdgree[i] = l

        total_hh_size = {}
        for i in range(0, self.sizes[3]):
            l = len(self.table.loc[self.table['hhsize'] == i])
            total_hh_size[i] = l

        return total_female, total_male, total_age, total_age_f, total_age_m, total_hdgree, total_hh_size

    def gather_marginals(self):
        total_female, total_male, total_age, total_age_f, total_age_m, total_hdgree, total_hh_size = self.create_marginals()

        # get marginal by sex, by age, by agebysex, hdgree, hhsize
        # 0:F 1:M
        marginal_sex = np.array([total_age_f, total_male])
        # 0:0-4y ... 17: 85+
        marginal_age = np.array(list(total_age.values()))
        # 0: F age, 1: M age
        marginal_age_by_sex = np.array([list(total_age_f.values()), list(total_age_m.values())])
        # 0: no, 1:secondary, 2: university
        marginal_hdgree = np.array(list(total_hdgree.values()))
        # 0: 1; 1: 2; 2: 3; 3: 4; 4: 5+
        marginal_hh_size = np.array(list(total_hh_size.values()))

        return marginal_sex, marginal_age, marginal_age_by_sex, marginal_hdgree, marginal_hh_size

    def ipf(self):
        seed = self.create_seed()

        i0 = np.array([0])
        i1 = np.array([1])
        i2 = np.array([0, 1])
        i3 = np.array([2])
        i4 = np.array([3])

        marginal_sex, marginal_age, marginal_age_by_sex, marginal_hdgree, marginal_hh_size = self.gather_marginals()

        p = humanleague.ipf(seed, [i0, i1, i2, i3, i4],
                            [marginal_sex.astype(float), marginal_age.astype(float),
                             marginal_age_by_sex.astype(float), marginal_hdgree.astype(float),
                             marginal_hh_size.astype(float)])

        return p[0]

    def create_probability(self, dim, p):
        arr = np.empty(dim)
        for i in range(dim):
            result = self.give_the_good_result(i=i, dim=dim, p=p)
            percent = result / p.sum()
            if dim == 3:
                arr[i] = percent.round(2)
            elif dim == 5:
                arr[i] = percent.round(1)
            else:
                arr[i] = percent.round(3)
        return arr

    def give_the_good_result(self, i, dim, p):
        if dim == 2:
            return p[i, :].sum()
        elif dim == 18:
            return p[:, i, :].sum()
        elif dim == 3:
            return p[:, :, i, :].sum()
        else:
            return p[:, :, :, i].sum()

    def sample_from_ipf(self, pop_num):
        gender = self.create_probability(dim=2, p=self.p)
        sel_gender = np.random.choice(len(gender), pop_num, True, gender)

        agegrp = self.create_probability(dim=18, p=self.p)
        sel_agegrp = np.random.choice(len(agegrp), pop_num, True, agegrp)

        hdgree = self.create_probability(dim=3, p=self.p)
        sel_hdgree = np.random.choice(len(hdgree), pop_num, True, hdgree)

        hsize = self.create_probability(dim=5, p=self.p)
        sel_hsize = np.random.choice(len(hsize), pop_num, True, hsize)

        chunk = pd.DataFrame(
            columns=['sex', "agegrp", "hdgree", "hhsize"])
        syn_inds = pd.DataFrame(columns=['sex', "agegrp", "hdgree", "hhsize"])
        chunk.sex = sel_gender
        chunk.agegrp = sel_agegrp
        chunk.hdgree = sel_hdgree
        chunk.hhsize = sel_hsize
        syn_inds = pd.concat([syn_inds, chunk], ignore_index=True)

        syn_inds.to_csv(os.path.join(PROJECT_PATH, "data", "/ipf.csv"), index=False)
        return syn_inds




