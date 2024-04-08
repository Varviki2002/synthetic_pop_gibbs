import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Validation:
    def __init__(self, ipf):
        self.ipf_data = ipf
        self.gibbs_data = pd.read_csv("../data/samples.csv")
        self.census = pd.read_csv("../data/synthetic_pop_y_2021.csv")

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
            labels = [f'{(v.get_height() / 1000000):.3f}m' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')


def main():
    gender_1 = {"sex": ["female", "male"]}
    gender_2 = {"sex": ["female", "male"]}

    age_group_1 = {"agegrp": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "40-44", "45-49", "50-59",
                              "60-64", "65-69", "70-74", "75-79" "80-84", "85-89", "90-94", "95-99", "100"]}
    age_group_2 = {"agegrp": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "40-44", "45-49", "50-59",
                              "60-64", "65-69", "70-74", "75-79" "80-84", "85-89", "90-94", "95-99", "100"]}

    hdgree_1 = {"hdgree": ["no", "secondary", "university"]}
    hdgree_2 = {"hdgree": ["no", "secondary", "university"]}

    hhsize_1 = {"hhsize": ["1", "2", "3", "4", "5+"]}
    hhsize_2 = {"hhsize": ["1", "2", "3", "4", "5+"]}

    validation = Validation(ipf=None)
    validation.plot_figures(col_name="sex", dict_1=gender_1, dict_2=gender_2)
    validation.plot_figures(col_name="agegrp", dict_1=age_group_1, dict_2=age_group_2)
    validation.plot_figures(col_name="hdgree", dict_1=hdgree_1, dict_2=hdgree_2)
    validation.plot_figures(col_name="hhsize", dict_1=hhsize_1, dict_2=hhsize_2)
    plt.show()


if __name__ == "__main__":
    main()