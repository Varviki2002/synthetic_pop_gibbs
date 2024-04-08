import itertools
import pandas as pd
# import gdown

import warnings
warnings.filterwarnings('ignore')


class CondCreat:
    def __init__(self, table, evidence):
        self.table = table
        self.evidence = evidence
        self.cross_1 = None
        self.cross_2 = None
        self.cross_3 = None
        self.cross_4 = None

    @staticmethod
    def calculate_cond_table(lista, df):
        new_df = df.copy()
        for tup in itertools.product(*lista):
            conditional = new_df[tup] / sum(new_df[tup])
            new_df[tup] = conditional
        return new_df

    def create_cross_tables(self):
        self.cross_1 = pd.crosstab([self.table.agegrp],
                                   [self.table.sex, self.table.hhsize, self.table.hdgree], dropna=False)
        self.cross_1.fillna(0, inplace=True)
        self.cross_2 = pd.crosstab([self.table.sex],
                                   [self.table.agegrp, self.table.hhsize, self.table.hdgree], dropna=False)
        self.cross_2.fillna(0, inplace=True)
        self.cross_3 = pd.crosstab([self.table.hdgree],
                                   [self.table.agegrp, self.table.sex, self.table.hhsize], dropna=False)
        self.cross_3.fillna(0, inplace=True)
        self.cross_4 = pd.crosstab([self.table.hhsize],
                                   [self.table.agegrp, self.table.sex, self.table.hdgree], dropna=False)
        self.cross_4.fillna(0, inplace=True)

    def create_conditional_tables(self):
        cardinalities_1 = [list(set(self.table["sex"])), list(set(self.table["hhsize"])),
                           list(set(self.table["hdgree"]))]
        cardinalities_2 = [list(set(self.table["agegrp"])), list(set(self.table["hhsize"])),
                           list(set(self.table["hdgree"]))]
        cardinalities_3 = [list(set(self.table["agegrp"])), list(set(self.table["sex"])),
                           list(set(self.table["hhsize"]))]
        cardinalities_4 = [list(set(self.table["agegrp"])), list(set(self.table["sex"])),
                           list(set(self.table["hdgree"]))]

        self.create_cross_tables()

        cond_1 = self.calculate_cond_table(cardinalities_1, self.cross_1)
        cond_2 = self.calculate_cond_table(cardinalities_2, self.cross_2)
        cond_3 = self.calculate_cond_table(cardinalities_3, self.cross_3)
        cond_4 = self.calculate_cond_table(cardinalities_4, self.cross_4)

        return cond_1, cond_2, cond_3, cond_4

    def save_cross_tables(self):
        cond_1, cond_2, cond_3, cond_4 = self.create_conditional_tables()

        cond_1.to_excel("../generated/cross_table_1.xlsx")
        cond_2.to_excel("../generated/cross_table_2.xlsx")
        cond_3.to_excel("../generated/cross_table_3.xlsx")
        cond_4.to_excel("../generated/cross_table_4.xlsx")


def main():
    # a file
    # url = "https://drive.google.com/uc?id=1BvkpS34abcnxJahdBOrY3h1BlKgMwm0S"
    # output = "synthetic_pop_y_2021.csv"
    # gdown.download(url, output)
    data = pd.read_csv("../data/synthetic_pop_y_2021.csv")

    evidence = {"agegrp": ["sex", "hhsize", "hdgree"], "sex": ["agegrp", "hhsize", "hdgree"],
                "hdgree": ["agegrp", "sex", "hhsize"], "hhsize": ["agegrp", "sex", "hdgree"]}

    cla = CondCreat(table=data, evidence=evidence)

    cla.save_cross_tables()


if __name__ == "__main__":
    main()
