import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from src import PROJECT_PATH
from src.create_conditionals import CondCreat
from src.dataloader import Downloader


class Validation:
    def __init__(self, creat_class: CondCreat, names, table):
        self.table = table
        self.names = names
        self.creat_class = creat_class
        self.ipf_10pr = pd.read_csv(os.path.join(PROJECT_PATH, "generated/ipf_05.csv"))
        self.ipf_10pr.drop([self.ipf_10pr.columns[0]], inplace=True, axis=1)
        # self.ipf_10pr.iloc[0]['agegrp'] = 0
        # self.ipf_10pr.iloc[1]['agegrp'] = 1
        # self.ipf_10pr.iloc[2]['agegrp'] = 2
        self.gibbs_10pr = pd.read_csv(os.path.join(PROJECT_PATH, "generated/samples_5percent.csv"))
        self.gibbs_data = pd.read_csv(os.path.join(PROJECT_PATH, "generated/samples_5percent.csv"))
        self.census = self.creat_class.table
        self.gibbs_partial_1 = pd.read_csv(os.path.join(PROJECT_PATH, "generated/samples_partial_05.csv"))

        self.cross_census = self.creat_class.create_cross_tables(
            table=self.table, name_list=self.names,
            idx=0, value=None, aggfunc=None)
        self.cross_gibbs = self.creat_class.create_cross_tables(
            table=self.gibbs_data, name_list=self.names, idx=0,
            value=None, aggfunc=None)
        self.cross_partial_1 = self.creat_class.create_cross_tables(
            table=self.gibbs_partial_1, name_list=self.names,
            idx=0, value=None, aggfunc=None)
        self.cross_ipf = self.creat_class.create_cross_tables(table=self.ipf_10pr,
                                                              name_list=self.names,
                                                              idx=0, value=None,
                                                              aggfunc=None)

    def create_columns(self, col_name, dict_1, dict_2):
        tab_1 = self.table[col_name].value_counts(sort=False)
        tab_2 = self.gibbs_data[col_name].value_counts(sort=False)

        dict_1["count"] = tab_1
        dict_2["count"] = tab_2

        df_1 = pd.DataFrame(dict_1)
        df_2 = pd.DataFrame(dict_2)

        df_1["ds"] = "Census"
        df_2["ds"] = "Simulation"
        dss = pd.concat([df_1, df_2])
        return dss

    def plot_figures(self, col_name, dict_1, dict_2, title):
        dss = self.create_columns(col_name=col_name, dict_1=dict_1, dict_2=dict_2)
        aspect = 2
        if col_name == "agegrp":
            aspect = 5
        g = sns.catplot(
            data=dss, kind="bar",
            x=col_name, y="count", hue="ds",
            palette="dark", alpha=.7, height=5, aspect=aspect
        )
        g.despine(left=True)
        g.set_axis_labels("", "Number of people")
        g.legend.set_title("")
        g.fig.suptitle(title, fontsize="small")
        ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]

        # iterate through the axes containers
        for c in ax.containers:
            labels = [f"{(v.get_height() / 1000):.2f}k" for v in c]
            ax.bar_label(c, labels=labels, label_type="edge")

    @staticmethod
    def linear_regression(x, y):
        x_reg = list(x.values.ravel())
        y_reg = list(y.values.ravel())

        x_reg = sm.add_constant(x_reg)

        model = sm.OLS(y_reg, x_reg).fit()
        # predictions = model.predict(x_reg)

        return model

    def plot_lin_regression(self, x, y, xlabel, title=None):
        model = self.linear_regression(x=x, y=y)
        x_reg = list(x.values.ravel())
        y_reg = list(y.values.ravel())
        fig, ax = plt.subplots()
        plt.scatter(x_reg, y_reg)
        sm.graphics.abline_plot(model_results=model, color="red", ax=ax)
        plt.legend(["people with different characteristics", f"y = {model.params[1].round(5)}x"])
        plt.text(0, 9000, f"R^2:{model.rsquared.round(5)}",
                 bbox=dict(facecolor="blue", alpha=0.2), fontsize="xx-large")
        # plt.text(0, 170000, f"y = {model.params[1].round(5)}x", bbox=dict(facecolor="blue", alpha=0.2))
        plt.xlabel(xlabel=xlabel)
        plt.ylabel("Census")
        plt.title(title)
        plt.show()

    @staticmethod
    def calculate_pearson_r(census, simulation):
        x = list(census.values.ravel())
        y = list(simulation.values.ravel())
        r = stats.pearsonr(x, y)
        print(f"The Pearson correlation coefficient is: {r.statistic}.")
        return r

    @staticmethod
    def calculate_nrmse(census, simulation):
        x = list(census.values.ravel())
        y = list(simulation.values.ravel())
        mse = mean_squared_error(x, y)

        rmse = np.sqrt(mse)

        nrmse = rmse/(max(x)-min(x))
        print(f"The Normalised Standardised Root Mean Square Error is: {nrmse}.")
        return nrmse

    @staticmethod
    def calculate_srmse(census, simulation):
        x = list(census.values.ravel())
        y = list(simulation.values.ravel())
        mse = mean_squared_error(x, y)

        rmse = np.sqrt(mse)

        denominator = np.sum(x) / len(x)
        srmse = rmse / denominator
        print(f"The Standardised Root Mean Square Error is: {srmse}.")
        return srmse

    @staticmethod
    def calculate_rae(census, simulation):
        x = census.values.ravel()
        y = simulation.values.ravel()
        rae = np.abs(x - y)/x
        rae = np.nan_to_num(rae)
        print(f"The Relative Absolute Error is: {rae}.")
        return rae

    def run_calculations(self, census, simulation):
        r = self.calculate_pearson_r(census=census, simulation=simulation)
        nrmse = self.calculate_nrmse(census=census, simulation=simulation)
        rae = self.calculate_rae(census=census, simulation=simulation)
        srmse = self.calculate_srmse(census=census, simulation=simulation)
        return r, nrmse, rae, srmse


def main():
    data = Downloader.read_data(
        file=os.path.join(PROJECT_PATH, "data/Census_2016_Individual_PUMF.dta"))
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
    gender_1 = {"Sex": ["female", "male"]}
    gender_2 = {"Sex": ["female", "male"]}
    age_group_1 = {"agegrp": ["0-4", "5-9", "10-14", "15-19", "20-24",
                              "25-29", "30-34", "40-44", "45-49", "50-59",
                              "60-64", "65-69", "70-74", "75-79" "80-84",
                              "85-89", "90-94", "95-99", "100"]}
    age_group_2 = {"agegrp": ["0-4", "5-9", "10-14", "15-19", "20-24",
                              "25-29", "30-34", "40-44", "45-49", "50-59",
                              "60-64", "65-69", "70-74", "75-79" "80-84",
                              "85-89", "90-94", "95-99", "100"]}
    hdgree_1 = {"hdgree": ["nincs", "középiskola", "egyetem"]}
    hdgree_2 = {"hdgree": ["nincs", "középiskola", "egyetem"]}
    hhsize_1 = {"hhsize": ["1", "2", "3", "4", "5+"]}
    hhsize_2 = {"hhsize": ["1", "2", "3", "4", "5+"]}
    validation = Validation(
        creat_class=CondCreat(table=data, full_evidence=evidence,
                              partial_evidence=evidence_partial_1,
                              full_names=names, save=False, partial_1=partial_1, size=0.2),
        names=names, table=data)
    validation.plot_figures(col_name="Sex", dict_1=gender_1, dict_2=gender_2, title="Gender")
    validation.plot_figures(col_name="agegrp", dict_1=age_group_1, dict_2=age_group_2, title="Age groups")
    validation.plot_figures(col_name="hdgree", dict_1=hdgree_1, dict_2=hdgree_2, title="Highest degrees")
    validation.plot_figures(col_name="hhsize", dict_1=hhsize_1, dict_2=hhsize_2, title="Household sizes")
    plt.show()
    validation.plot_lin_regression(
        x=validation.cross_gibbs, y=validation.cross_census,
        xlabel="Simulation", title="Full conditionals")
    print("Gibbs-full:")
    r, nrmse, rae, srmse = validation.run_calculations(
        census=validation.cross_census,
        simulation=validation.cross_gibbs)
    print(min(rae), max(rae), np.mean(rae))
    validation.plot_lin_regression(
        x=validation.cross_ipf, y=validation.cross_census,
        xlabel="Simulation", title="IPF algorithm")
    validation.plot_lin_regression(
        x=validation.cross_gibbs, y=validation.cross_partial_1,
        xlabel="Simulation", title="Partial_1")
    print("IPF:")
    r, nrmse, rae, srmse = validation.run_calculations(
        census=validation.cross_census,
        simulation=validation.cross_ipf)
    print(min(rae), max(rae), np.mean(rae))
    print("Gibbs-partial")
    r, nrmse, rae, srmse = validation.run_calculations(
        census=validation.cross_census,
        simulation=validation.cross_partial_1)
    print(min(rae), max(rae), np.mean(rae))



if __name__ == "__main__":
    main()
