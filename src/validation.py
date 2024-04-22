import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from src import PROJECT_PATH
from src.create_synth_pop import CondCreat
from src.dataloader import Downloader


class Validation:
    def __init__(self, creat_class: CondCreat, names):
        self.names = names
        self.creat_class = creat_class
        self.ipf_10pr = pd.read_csv(os.path.join(PROJECT_PATH, "data/ipf_10pr.csv"))
        # self.ipf_data.drop([self.ipf_data.columns[0]], inplace=True, axis=1)
        self.gibbs_10pr = pd.read_csv(os.path.join(PROJECT_PATH, "generated/samples_10pr.csv"))
        self.gibbs_data = pd.read_csv(os.path.join(PROJECT_PATH, "generated/samples.csv"))
        self.census = self.creat_class.table
        self.gibbs_partial_1 = pd.read_csv(os.path.join(PROJECT_PATH, "generated/samples_partial.csv"))

        self.cross_census = self.creat_class._create_cross_tables(table=self.creat_class.table, name_list=self.names,
                                                                  idx=0, value=None, aggfunc=None)
        self.cross_gibbs = self.creat_class._create_cross_tables(table=self.gibbs_data, name_list=self.names, idx=0,
                                                                 value=None, aggfunc=None)
        self.cross_partial_1 = self.creat_class._create_cross_tables(table=self.gibbs_partial_1, name_list=self.names,
                                                                     idx=0, value=None, aggfunc=None)
        # self.cross_ipf = self.creat_class.create_cross_tables(self.gibbs_data, self.names, 0, None, "count")

    def create_columns(self, col_name, dict_1, dict_2):
        tab_1 = self.census[col_name].value_counts(sort=False)
        tab_2 = self.gibbs_data[col_name].value_counts(sort=False)

        dict_1["count"] = tab_1
        dict_2["count"] = tab_2

        df_1 = pd.DataFrame(dict_1)
        df_2 = pd.DataFrame(dict_2)

        df_1["ds"] = "Census"
        df_2["ds"] = "Simulation"
        dss = pd.concat([df_1, df_2])
        return dss

    def plot_figures(self, col_name, dict_1, dict_2):
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
        ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]

        # iterate through the axes containers
        for c in ax.containers:
            labels = [f"{(v.get_height() / 10000):.3f}m" for v in c]
            ax.bar_label(c, labels=labels, label_type="edge")

    @staticmethod
    def linear_regression(x, y):
        X = list(x.values.ravel())
        Y = list(y.values.ravel())

        X = sm.add_constant(X)

        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X)

        return model

    def plot_lin_regression(self, x, y, xlabel, title=None):
        model = self.linear_regression(x=x, y=y)
        X = list(x.values.ravel())
        Y = list(y.values.ravel())
        fig, ax = plt.subplots()
        plt.scatter(X, Y)
        sm.graphics.abline_plot(model_results=model, color="red", ax=ax)
        plt.legend(["people with different characteristics", f"y = {model.params[1].round(5)}x"])
        plt.text(0, 9000, f"R^2:{model.rsquared.round(5)}", bbox=dict(facecolor="blue", alpha=0.2))
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
        return r, nrmse, rae

