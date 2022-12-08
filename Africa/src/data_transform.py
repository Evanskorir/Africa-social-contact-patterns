import numpy as np

from dataloader import DataLoader
from simulation import Simulation
from data_transformer import Contacts


class DataTransformer:
    def __init__(self, susc: float = 1.0, base_r0: float = 2.2):
        self.susc = susc
        self.base_r0 = base_r0

        self.data = DataLoader()
        self.upper_tri_indexes = np.triu_indices(6)
        self.country_names = list(self.data.age_data.keys())
        self.country_matrix = []

        self.data_all_dict = dict()
        self.data_cm_d2pca_col = []
        self.data_cm_d2pca_row = []
        self.data_cm_1dpca = []
        self.data_cm_pca = []
        self.indicator_data = []
        self.age_vector = []
        self.contacts = np.array([])
        self.age_group = np.array([])

        self.get_data_for_clustering()

    def get_data_for_clustering(self):
        for country in self.country_names:
            age_vector = self.data.age_data[country]["age"].reshape((-1, 1))

            contact_home = self.data.contact_data[country]["HOME"]
            contact_school = self.data.contact_data[country]["SCHOOL"]
            contact_work = self.data.contact_data[country]["WORK"]
            contact_other = self.data.contact_data[country]["OTHER"]
            #contact_matrix = contact_home + contact_school + contact_work + contact_other
            contact_matrix = self.contacts

            susceptibility = np.array([1.0] * 6)
            susceptibility[:4] = self.susc
            simulation = Simulation(data=self.data, base_r0=self.base_r0,
                                    contact_matrix=contact_matrix,
                                    #age_vector=age_vector,
                                    age_vector=self.age_group,
                                    susceptibility=susceptibility)
            # Create dictionary with all necessary data
            self.data_all_dict.update(
                {country: {"beta": simulation.beta,
                           "age_vector": age_vector,
                           "contact_full": contact_matrix,
                           "contact_home": contact_home,
                           "contact_school": contact_school,
                           "contact_work": contact_work,
                           "contact_other": contact_other
                           }
                 })
            self.age_vector = age_vector
            self.country_matrix = contact_matrix
            # Create separated data structure for (2D)^2 PCA
            self.data_cm_d2pca_col.append(
                simulation.beta * contact_matrix)
            self.data_cm_d2pca_row.append(
                simulation.beta * contact_matrix.T
            )
            # create data for the indicators
            self.indicator_data = self.data.indicators_data

            # Create separated data structure for 1D PCA
            self.data_cm_1dpca.append(
                simulation.beta * contact_matrix[self.upper_tri_indexes])
        self.data_cm_1dpca = np.array(self.data_cm_1dpca)
        # Final shape of the np.ndarrays: (624, 16)
        self.data_cm_d2pca_col = np.vstack(self.data_cm_d2pca_col)
        self.data_cm_d2pca_row = np.vstack(self.data_cm_d2pca_row)


def main():
    data_tr = DataTransformer()
    x = Contacts()
    print(data_tr.country_matrix)


if __name__ == "__main__":
    main()





