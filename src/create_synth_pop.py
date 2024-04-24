import os

import itertools
import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore")

from src import PROJECT_PATH
from src.dataloader import Downloader


class CondCreat:
    def __init__(self, table, full_evidence, partial_evidence, full_names, partial_1, save):
        self.table = self._create_sample_table(table)
        self.full_evidence = full_evidence
        self.partial_evidence = partial_evidence
        self.full_names = full_names
        self.partial_names_1 = partial_1
        self.full_cross_1 = self._create_cross_tables(table=self.table, name_list=self.full_names, idx=0,
                                                      value=self.table["weight"], aggfunc="sum")
        self.full_cross_2 = self._create_cross_tables(table=self.table, name_list=self.full_names, idx=1,
                                                      value=self.table["weight"], aggfunc="sum")
        self.full_cross_3 = self._create_cross_tables(table=self.table, name_list=self.full_names, idx=2,
                                                      value=self.table["weight"], aggfunc="sum")
        self.full_cross_4 = self._create_cross_tables(table=self.table, name_list=self.full_names, idx=3,
                                                      value=self.table["weight"], aggfunc="sum")
        self.full_cross_5 = self._create_cross_tables(table=self.table, name_list=self.full_names, idx=4,
                                                      value=self.table["weight"], aggfunc="sum")
        self.full_cross_6 = self._create_cross_tables(table=self.table, name_list=self.full_names, idx=5,
                                                      value=self.table["weight"], aggfunc="sum")

        self.partial_cross_1 = self._create_cross_tables(table=self.table, name_list=self.partial_names_1,
                                                         idx=0, value=self.table["weight"], aggfunc="sum")

        if save is True:
            self._save_full_cond_tables()

    @staticmethod
    def _create_sample_table(table):
        index = table.index.to_list()
        choice = list(np.random.choice(index, size=183320, replace=False))
        table = table[table.index.isin(choice)]
        return table

    @staticmethod
    def calculate_cond_table(lista, df):
        new_df = df.copy()
        for tup in itertools.product(*lista):
            conditional = new_df[tup] / sum(new_df[tup])
            new_df[tup] = conditional
        return new_df

    @staticmethod
    def _create_cross_tables(table, name_list, idx, value, aggfunc):
        index = name_list[idx]
        column = [table[v] for v in name_list if name_list[idx] != v]

        cross = pd.crosstab(index=[table[index]], columns=[v for v in column], values=value,
                            aggfunc=aggfunc, dropna=False)
        cross.fillna(value=0, inplace=True)

        return cross

    def create_cardinality(self, name_list, idx, full: bool):
        if full is True:
            cardinality = [list(set(self.table[v])) for v in self.full_evidence[name_list[idx]]]
        else:
            cardinality = [list(set(self.table[v])) for v in self.partial_evidence[name_list[idx]]]
        return cardinality

    def create_conditional_tables(self):
        cardinalities_1 = self.create_cardinality(name_list=self.full_names, idx=0, full=True)
        cardinalities_2 = self.create_cardinality(name_list=self.full_names, idx=1, full=True)
        cardinalities_3 = self.create_cardinality(name_list=self.full_names, idx=2, full=True)
        cardinalities_4 = self.create_cardinality(name_list=self.full_names, idx=3, full=True)
        cardinalities_5 = self.create_cardinality(name_list=self.full_names, idx=4, full=True)
        cardinalities_6 = self.create_cardinality(name_list=self.full_names, idx=5, full=True)
        cardinalities_partial_1 = self.create_cardinality(name_list=self.partial_names_1, idx=0,
                                                          full=False)

        cond_1 = self.calculate_cond_table(lista=cardinalities_1, df=self.full_cross_1)
        cond_2 = self.calculate_cond_table(lista=cardinalities_2, df=self.full_cross_2)
        cond_3 = self.calculate_cond_table(lista=cardinalities_3, df=self.full_cross_3)
        cond_4 = self.calculate_cond_table(lista=cardinalities_4, df=self.full_cross_4)
        cond_5 = self.calculate_cond_table(lista=cardinalities_5, df=self.full_cross_5)
        cond_6 = self.calculate_cond_table(lista=cardinalities_6, df=self.full_cross_6)

        cond_partial_1 = self.calculate_cond_table(lista=cardinalities_partial_1, df=self.partial_cross_1)

        return cond_1, cond_2, cond_3, cond_4, cond_5, cond_6, cond_partial_1

    def _save_full_cond_tables(self):
        cond_1, cond_2, cond_3, cond_4, cond_5, cond_6, cond_partial_1 = self.create_conditional_tables()

        data_generated = os.path.join(PROJECT_PATH, "generated")

        cond_1.to_csv(os.path.join(data_generated, "full_cross_table_1.csv"))
        cond_2.to_csv(os.path.join(data_generated, "full_cross_table_2.csv"))
        cond_3.to_csv(os.path.join(data_generated, "full_cross_table_3.csv"))
        cond_4.to_csv(os.path.join(data_generated, "full_cross_table_4.csv"))
        cond_5.to_csv(os.path.join(data_generated, "full_cross_table_5.csv"))
        cond_6.to_csv(os.path.join(data_generated, "full_cross_table_6.csv"))
        cond_partial_1.to_csv(os.path.join(data_generated, "partial_cross_1.csv"))

def main():
    data = Downloader.read_data(file=os.path.join(PROJECT_PATH, "data/Census_2016_Individual_PUMF.dta"))
    names = ["agegrp", "Sex", "hdgree", "lfact", "TotInc", "hhsize"]
    partial_1 = ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"]

    evidence = {"agegrp": ["Sex", "hdgree", "lfact", "TotInc", "hhsize"],
                "Sex": ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"],
                "hdgree": ["agegrp", "Sex", "lfact", "TotInc", "hhsize"],
                "lfact": ["agegrp", "Sex", "hdgree", "TotInc", "hhsize"],
                "TotInc": ["agegrp", "Sex", "hdgree", "lfact", "hhsize"],
                "hhsize": ["agegrp", "Sex", "hdgree", "lfact", "TotInc"]}

    evidence_partial_1 = {"agegrp": ["hdgree", "lfact", "TotInc", "hhsize"],
                          "Sex": ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"],
                          "hdgree": ["agegrp", "Sex", "lfact", "TotInc", "hhsize"],
                          "lfact": ["agegrp", "Sex", "hdgree", "TotInc", "hhsize"],
                          "TotInc": ["agegrp", "Sex", "hdgree", "lfact", "hhsize"],
                          "hhsize": ["agegrp", "Sex", "hdgree", "lfact", "TotInc"]}
    cla = CondCreat(table=data, full_evidence=evidence, partial_evidence=evidence_partial_1, full_names=names,
                    save=True, partial_1=partial_1)
if __name__ == "__main__":
    main()
