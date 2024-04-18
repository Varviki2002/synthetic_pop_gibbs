import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from src import PROJECT_PATH
from src.create_synth_pop import CondCreate
from src.dataloader import Downloader


class Validation:
    def __init__(self, create_class: CondCreate, names):
        self.names = names
        self.create_class = create_class
        self.ipf_10pr = pd.read_csv(
            os.path.join(PROJECT_PATH, "data/ipf_10pr.csv")
        )
        # self.ipf_data.drop([self.ipf_data.columns[0]], inplace=True, axis=1)
        self.gibbs_10pr = pd.read_csv(
            os.path.join(PROJECT_PATH, "generated/samples_10pr.csv")
        )
        self.gibbs_data = pd.read_csv(
            os.path.join(PROJECT_PATH, "generated/samples.csv")
        )
        self.census = self.create_class.table
        self.gibbs_partial_1 = pd.read_csv(
            os.path.join(PROJECT_PATH, "generated/samples_partial.csv")
        )

        self.cross_census = self.create_class.create_cross_tables(
            table=self.create_class.table, name_list=self.names,
            idx=0, value=None, aggfunc=None)
        self.cross_gibbs = self.create_class.create_cross_tables(
            table=self.gibbs_data, name_list=self.names, idx=0,
            value=None, aggfunc=None)
        self.cross_partial_1 = self.create_class.create_cross_tables(
            table=self.gibbs_partial_1, name_list=self.names,
            idx=0, value=None, aggfunc=None)
        # self.cross_ipf = self.creat_class.create_cross_tables(self.gibbs_data, self.names, 0, None, "count")

    def create_columns(self, col_name, dict_1, dict_2):
        tab_1 = self.census[col_name].value_counts(sort=False)
        tab_2 = self.gibbs_data[col_name].value_counts(sort=False)

        dict_1["count"] = tab_1
        dict_2["count"] = tab_2

        df_1 = pd.DataFrame(dict_1)
        df_2 = pd.DataFrame(dict_2)

        df_1['ds'] = 'Census'
        df_2['ds'] = 'Simulation'
        dss = pd.concat([df_1, df_2])
        return dss

    def plot_figures(self, col_name, dict_1, dict_2):
        dss = self.create_columns(col_name=col_name, dict_1=dict_1, dict_2=dict_2)
        aspect = 2
        if col_name == 'agegrp':
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
            labels = [f'{(v.get_height() / 10000):.3f}m' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')

    @staticmethod
    def linear_regression(x, y):
        x_lin = list(x.values.ravel())
        y_lin = list(y.values.ravel())

        x_lin = sm.add_constant(x_lin)

        model = sm.OLS(y_lin, x_lin).fit()
        predictions = model.predict(x_lin)
        return model

    def plot_lin_regression(self, x, y, xlabel, title=None):
        model = self.linear_regression(x=x, y=y)
        x_lin = list(x.values.ravel())
        y_lin = list(y.values.ravel())
        fig, ax = plt.subplots()
        plt.scatter(x_lin, y_lin)
        sm.graphics.abline_plot(model_results=model, color='red', ax=ax)
        plt.legend(["people with different characteristics",
                    f"y = {model.params[1].round(5)}x"])
        plt.text(0, 9000,
                 f"R^2:{model.rsquared.round(5)}",
                 bbox=dict(facecolor='blue', alpha=0.2))
        # plt.text(0, 170000, f"y = {model.params[1].round(5)}x", bbox=dict(facecolor='blue', alpha=0.2))
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


def main():
    data = Downloader.read_data(
        file=os.path.join(PROJECT_PATH, "data/Census_2016_Individual_PUMF.dta")
    )
    names = ["agegrp", "Sex", "hdgree", "lfact", "TotInc", "hhsize"]
    partial_1 = ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"]

    evidence = {
        "agegrp": ["Sex", "hdgree", "lfact", "TotInc", "hhsize"],
        "Sex": ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"],
        "hdgree": ["agegrp", "Sex", "lfact", "TotInc", "hhsize"],
        "lfact": ["agegrp", "Sex", "hdgree", "TotInc", "hhsize"],
        "TotInc": ["agegrp", "Sex", "hdgree", "lfact", "hhsize"],
        "hhsize": ["agegrp", "Sex", "hdgree", "lfact", "TotInc"]
    }

    evidence_partial_1 = {
        "agegrp": ["hdgree", "lfact", "TotInc", "hhsize"],
        "Sex": ["agegrp", "hdgree", "lfact", "TotInc", "hhsize"],
        "hdgree": ["agegrp", "Sex", "lfact", "TotInc", "hhsize"],
        "lfact": ["agegrp", "Sex", "hdgree", "TotInc", "hhsize"],
        "TotInc": ["agegrp", "Sex", "hdgree", "lfact", "hhsize"],
        "hhsize": ["agegrp", "Sex", "hdgree", "lfact", "TotInc"]
    }
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

    hdgree_1 = {"hdgree": ["no", "secondary", "university"]}
    hdgree_2 = {"hdgree": ["no", "secondary", "university"]}

    hhsize_1 = {"hhsize": ["1", "2", "3", "4", "5+"]}
    hhsize_2 = {"hhsize": ["1", "2", "3", "4", "5+"]}

    validation = Validation(
        create_class=CondCreate(
            table=data, full_evidence=evidence,
            partial_evidence=evidence_partial_1,
            full_names=names, save=False, parial_1=partial_1),
        names=names
    )
    validation.plot_figures(
        col_name="Sex", dict_1=gender_1, dict_2=gender_2
    )
    validation.plot_figures(
        col_name="agegrp", dict_1=age_group_1, dict_2=age_group_2
    )
    validation.plot_figures(
        col_name="hdgree", dict_1=hdgree_1, dict_2=hdgree_2
    )
    validation.plot_figures(
        col_name="hhsize", dict_1=hhsize_1, dict_2=hhsize_2
    )
    plt.show()

    validation.plot_lin_regression(
        x=validation.cross_gibbs, y=validation.cross_census,
        xlabel="Simulation", title="Full conditionals"
    )

    r, nrmse, rae = validation.run_calculations(
        census=validation.cross_census,
        simulation=validation.cross_gibbs
    )

    validation.plot_lin_regression(
        x=validation.cross_partial_1,
        y=validation.cross_census,
        xlabel="Simulation", title="Partial_1"
    )


if __name__ == "__main__":
    main()
