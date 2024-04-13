

class DataManipulator:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def map_age_grp(df_indiv):
        df_indiv.loc[((df_indiv["agegrp"] == 2) & (df_indiv["agegrp"] == 3)), "agegrp"] = 1
        df_indiv.loc[((df_indiv["agegrp"] == 4) & (df_indiv["agegrp"] == 5)), "agegrp"] = 2
        df_indiv.loc[((df_indiv["agegrp"] == 6) & (df_indiv["agegrp"] == 7)), "agegrp"] = 3
        for i in range(4, 18):
            df_indiv.loc[df_indiv["agegrp"] == i + 4, "agegrp"] = i
        df_indiv = df_indiv.loc[df_indiv["agegrp"] != 88]
        return df_indiv

    # Map hdgree to 4 classes
    @staticmethod
    def map_hdgree(df_indiv):
        df_indiv.loc[df_indiv["hdgree"] == 88, "hdgree"] = 0
        df_indiv.loc[df_indiv["hdgree"] == 99, "hdgree"] = 0
        df_indiv.loc[df_indiv["hdgree"] > 2, "hdgree"] = 2
        df_indiv.loc[df_indiv["hdgree"] == 1, "hdgree"] = 0
        df_indiv.loc[df_indiv["hdgree"] == 2, "hdgree"] = 1
        return df_indiv

    # Map lfact to 3 classes
    @staticmethod
    def map_lfact(df_indiv):
        df_indiv.loc[df_indiv["lfact"] == 1, "lfact"] = 0
        df_indiv.loc[df_indiv["lfact"] == 2, "lfact"] = 0
        df_indiv.loc[((df_indiv["lfact"] > 2) & (df_indiv["lfact"] < 11)), "lfact"] = 1
        df_indiv.loc[df_indiv["lfact"] > 10, "lfact"] = 2
        return df_indiv

    # Map hhsize to 5 classes
    @staticmethod
    def map_hhsize(df_indiv):
        df_indiv.loc[df_indiv["hhsize"] == 1, "hhsize"] = 0
        df_indiv.loc[df_indiv["hhsize"] == 8, "hhsize"] = 0
        df_indiv.loc[df_indiv["hhsize"] == 2, "hhsize"] = 1
        df_indiv.loc[df_indiv["hhsize"] == 3, "hhsize"] = 2
        df_indiv.loc[df_indiv["hhsize"] == 4, "hhsize"] = 3
        df_indiv.loc[df_indiv["hhsize"] > 4, "hhsize"] = 4
        return df_indiv

    # Map totinc to 4 classes
    @staticmethod
    def map_totinc(df_indiv):
        df_indiv = df_indiv.loc[df_indiv["TotInc"] != 88888888]
        df_indiv.loc[df_indiv["TotInc"] == 99999999, "TotInc"] = 0
        df_indiv.loc[df_indiv["TotInc"] < 20000, "TotInc"] = 0

        # for i in range(1, 10):
        #    df_indiv.loc[((df_indiv["TotInc"] >= 10000 * i) & (df_indiv["TotInc"] < 10000 * (i + 1))), "TotInc"] = 695 + i

        df_indiv.loc[((df_indiv["TotInc"] >= 20000) & (df_indiv["TotInc"] < 60000)), "TotInc"] = 1
        df_indiv.loc[((df_indiv["TotInc"] >= 60000) & (df_indiv["TotInc"] < 100000)), "TotInc"] = 2
        df_indiv.loc[df_indiv["TotInc"] >= 100000, "TotInc"] = 3

        return df_indiv

    @staticmethod
    def map_cfstat(df_indiv):
        df_indiv.loc[df_indiv["cfstat"] == 8, "cfstat"] = 7
        return df_indiv

    @staticmethod
    def map_sex(df_indiv):
        df_indiv.loc[df_indiv["Sex"] == 1, "Sex"] = 0
        df_indiv.loc[df_indiv["Sex"] == 2, "Sex"] = 1
