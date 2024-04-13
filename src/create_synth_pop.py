import os

import itertools
import pandas as pd

from src.dataloader import Downloader

import warnings

warnings.filterwarnings('ignore')

from src import PROJECT_PATH


class CondCreat:
    def __init__(self, table, evidence, full_names):
        self.table = table
        self.evidence = evidence
        self.full_names = full_names
        self.full_cross_1 = self.create_cross_tables(self.table, self.full_names, 0, value=self.table["weight"],
                                                     aggfunc="sum")
        self.full_cross_2 = self.create_cross_tables(self.table, self.full_names, 1, value=self.table["weight"],
                                                     aggfunc="sum")
        self.full_cross_3 = self.create_cross_tables(self.table, self.full_names, 2, value=self.table["weight"],
                                                     aggfunc="sum")
        self.full_cross_4 = self.create_cross_tables(self.table, self.full_names, 3, value=self.table["weight"],
                                                     aggfunc="sum")
        self.full_cross_5 = self.create_cross_tables(self.table, self.full_names, 4, value=self.table["weight"],
                                                     aggfunc="sum")
        self.full_cross_6 = self.create_cross_tables(self.table, self.full_names, 5, value=self.table["weight"],
                                                     aggfunc="sum")
        self.save_full_cond_tables()

        self.two_cross_1 = None
        self.two_cross_2 = None
        self.two_cross_3 = None
        self.two_cross_4 = None

    @staticmethod
    def calculate_cond_table(lista, df):
        new_df = df.copy()
        for tup in itertools.product(*lista):
            try:
                conditional = new_df[tup] / sum(new_df[tup])
                new_df[tup] = conditional
            except:
                conditional = new_df[tup] / sum(new_df[tup])
                new_df[tup] = conditional
        return new_df

    @staticmethod
    def create_cross_tables(table, name_list, idx, value, aggfunc):
        index = name_list[idx]
        column = [table[v] for v in name_list if name_list[idx] != v]

        cross = pd.crosstab(index=[table[index]], columns=[v for v in column], values=value, aggfunc=aggfunc, dropna=False)
        cross.fillna(0, inplace=True)

        return cross

    def create_cardinality(self, name_list, idx):
        cardinality = [list(set(self.table[v])) for v in self.evidence[name_list[idx]]]
        return cardinality

    def create_full_conditional_tables(self):
        cardinalities_1 = self.create_cardinality(self.full_names, idx=0)
        cardinalities_2 = self.create_cardinality(self.full_names, idx=1)
        cardinalities_3 = self.create_cardinality(self.full_names, idx=2)
        cardinalities_4 = self.create_cardinality(self.full_names, idx=3)
        cardinalities_5 = self.create_cardinality(self.full_names, idx=4)
        cardinalities_6 = self.create_cardinality(self.full_names, idx=5)

        cond_1 = self.calculate_cond_table(cardinalities_1, self.full_cross_1)
        cond_2 = self.calculate_cond_table(cardinalities_2, self.full_cross_2)
        cond_3 = self.calculate_cond_table(cardinalities_3, self.full_cross_3)
        cond_4 = self.calculate_cond_table(cardinalities_4, self.full_cross_4)
        cond_5 = self.calculate_cond_table(cardinalities_5, self.full_cross_5)
        cond_6 = self.calculate_cond_table(cardinalities_6, self.full_cross_6)

        return cond_1, cond_2, cond_3, cond_4, cond_5, cond_6

    def create_2_conditional_tables(self):
        cardinalities_1 = [list(set(self.table["sex"])), list(set(self.table["hhsize"])),
                           list(set(self.table["hdgree"]))]
        cardinalities_2 = [list(set(self.table["agegrp"])), list(set(self.table["hhsize"])),
                           list(set(self.table["hdgree"]))]
        cardinalities_3 = [list(set(self.table["agegrp"])), list(set(self.table["sex"])),
                           list(set(self.table["hhsize"]))]
        cardinalities_4 = [list(set(self.table["agegrp"])), list(set(self.table["sex"])),
                           list(set(self.table["hdgree"]))]

    def save_full_cond_tables(self):
        cond_1, cond_2, cond_3, cond_4, cond_5, cond_6 = self.create_full_conditional_tables()

        data_generated = os.path.join(PROJECT_PATH, "generated")

        cond_1.to_csv(os.path.join(data_generated, "full_cross_table_1.csv"))
        cond_2.to_csv(os.path.join(data_generated, "full_cross_table_2.csv"))
        cond_3.to_csv(os.path.join(data_generated, "full_cross_table_3.csv"))
        cond_4.to_csv(os.path.join(data_generated, "full_cross_table_4.csv"))
        cond_5.to_csv(os.path.join(data_generated, "full_cross_table_5.csv"))
        cond_6.to_csv(os.path.join(data_generated, "full_cross_table_6.csv"))

        # cond_1.to_excel(os.path.join(data_generated, "cross_table_1.xlsx"))
        # cond_2.to_excel(os.path.join(data_generated, "cross_table_2.xlsx"))
        # cond_3.to_excel(os.path.join(data_generated, "cross_table_3.xlsx"))
        # cond_4.to_excel(os.path.join(data_generated, "cross_table_4.xlsx"))



def main():
    # Downloader(gdrive_id="1fni7wudNdWjy5BsPpNXaxLGshyNtiapK", file_name="Census_2016_Individual_PUMF.dta")
    data = Downloader.read_data(file=os.path.join(PROJECT_PATH, "data/Census_2016_Individual_PUMF.dta"))


    names = ["agegrp", "Sex", "hdgree", "lfact", "TotInc", "hhsize"]

    evidence = {"agegrp": ["Sex", "hdgree", "lfact", "TotInc", "hhsize"],
                "Sex": ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"],
                "hdgree": ["agegrp", "Sex", "lfact", "TotInc", "hhsize"],
                "lfact": ["agegrp", "Sex", "hdgree", "TotInc", "hhsize"],
                "TotInc": ["agegrp", "Sex", "hdgree", "lfact", "hhsize"],
                "hhsize": ["agegrp", "Sex", "hdgree", "lfact", "TotInc"]}

    cla = CondCreat(table=data, evidence=evidence, full_names=names)


if __name__ == "__main__":
    main()
