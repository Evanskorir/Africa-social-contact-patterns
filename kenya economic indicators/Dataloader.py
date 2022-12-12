
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    The DataLoader object manages getting all data necessary for simulation

    Member variables:
        *._data_file     Data file paths
        *._data          Loaded data
    """

    def __init__(self):

        self.data = None
        self.data2 = None
        self.county_data_scaled = []
        self.county_names = []
        self.get_data()
        self.scale_data()

    def get_data(self):
        """
        Main function for social economic data loading
        :return: data
        """
        # use pandas to load the social_economic data

        data = pd.read_excel("/Users/user/PycharmProjects/kenya economic indicators/data.xls",
                             index_col=0)
        data2 = pd.read_excel("/Users/user/PycharmProjects/kenya economic indicators/data.xls")
        self.county_names = data2["County"]
        self.data = data
        self.data2 = data2
        return self.data, self.county_names

    def scale_data(self):
        """
        Scales the socio-economic data
        :param data:
        :return: ndarray of the scaled data
        """
        scaler = StandardScaler()
        county_data_scaled = scaler.fit_transform(self.data)
        self.county_data_scaled = county_data_scaled
        return self.county_data_scaled




