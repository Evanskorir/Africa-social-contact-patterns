import numpy as np

from dataloader import DataLoader
from data_transform import DataTransformer


class Home:
    def __init__(self, data_tr: DataTransformer, country_names: list):
        self.country_names = country_names
        self.data_tr = data_tr
        self.data = DataLoader()
        self.contact_home = dict()
        self.get_contact_home()

    def get_contact_home(self):
        for country in self.country_names:
            age_vector = self.data_tr.age_vector
            age_1 = age_vector[0]
            age_2 = age_vector[1] + age_vector[2]
            age_3 = age_vector[3]
            age_4 = age_vector[4]
            age_5 = age_vector[5] + age_vector[6] + age_vector[7] + age_vector[8] + age_vector[9] \
                    + age_vector[10] + age_vector[11] + age_vector[12]
            age_6 = age_vector[13] + age_vector[14] + age_vector[15]

            # first column
            # age group 0-4, 0-4
            col_0_0 = self.data.contact_data[country]["HOME"][0:1, 0:1]
            # age group 5-14, 0-4
            col_1_0 = (self.data.contact_data[country]["HOME"][1, 0] * age_vector[1]
                       + self.data.contact_data[country]["HOME"][2, 0] * age_vector[2]) / age_2
            # age group 15-19, 0-4
            col_2_0 = self.data.contact_data[country]["HOME"][3, 0]
            # age group 20-24, 0-4
            col_3_0 = self.data.contact_data[country]["HOME"][4:5, 0:1]
            # age group 25-64, 0-4
            col_4_0 = ((self.data.contact_data[country]["HOME"][5:6, 0:1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][6:7, 0:1] * age_vector[1]) +
                       (self.data.contact_data[country]["HOME"][7:8, 0:1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][8:9, 0:1] * age_vector[1]) +
                       (self.data.contact_data[country]["HOME"][9:10, 0:1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][10:11, 0:1] * age_vector[1]) +
                       (self.data.contact_data[country]["HOME"][11:12, 0:1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][12:13, 0:1] * age_vector[1])) / age_2
            # age group 65+, 0-4
            col_5_0 = ((self.data.contact_data[country]["HOME"][13:14, 0:1] * age_vector[1] +
                        (self.data.contact_data[country]["HOME"][14:15, 0:1] * age_vector[1]) +
                        self.data.contact_data[country]["HOME"][15:16, 0:1] * age_vector[1])) / age_2

            # second column
            # age group 0-4, 5-14
            col_0_1 = (self.data.contact_data[country]["HOME"][0, 1] * age_vector[1] +
                       self.data.contact_data[country]["HOME"][0, 2] * age_vector[2]) / age_2
            # age group 5-14, 5-14
            col_1_1 = ((self.data.contact_data[country]["HOME"][1, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][1, 2] * age_vector[2]) +
                       (self.data.contact_data[country]["HOME"][2, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][2, 2] * age_vector[2])) / age_2
            # age group 15-19, 5-14
            col_2_1 = (self.data.contact_data[country]["HOME"][3, 1] * age_vector[1] +
                       self.data.contact_data[country]["HOME"][3, 2] * age_vector[2]) / age_2
            # age group 20-24, 5-14
            col_3_1 = (self.data.contact_data[country]["HOME"][4, 1] * age_vector[1] +
                       self.data.contact_data[country]["HOME"][4, 2] * age_vector[2]) / age_2
            # age group 25-64, 5-14
            col_4_1 = ((self.data.contact_data[country]["HOME"][5, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][5, 2] * age_vector[2]) +
                       (self.data.contact_data[country]["HOME"][6, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][6, 2] * age_vector[2]) +
                       (self.data.contact_data[country]["HOME"][7, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][7, 2] * age_vector[2]) +
                       (self.data.contact_data[country]["HOME"][8, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][8, 2] * age_vector[2]) +
                       (self.data.contact_data[country]["HOME"][9, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][9, 2] * age_vector[2]) +
                       (self.data.contact_data[country]["HOME"][10, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][10, 2] * age_vector[2]) +
                       (self.data.contact_data[country]["HOME"][11, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][11, 2] * age_vector[2]) +
                       (self.data.contact_data[country]["HOME"][12, 1] * age_vector[1] +
                        self.data.contact_data[country]["HOME"][12, 2] * age_vector[2])) / age_2
            # age group 65+, 5-14
            col_5_1 = ((self.data.contact_data[country]["HOME"][13, 1] * age_vector[1] +
                       self.data.contact_data[country]["HOME"][13, 2] * age_vector[2]) + \
                      (self.data.contact_data[country]["HOME"][14, 1] * age_vector[1] +
                       self.data.contact_data[country]["HOME"][14, 2] * age_vector[2]) + \
                      (self.data.contact_data[country]["HOME"][15, 1] * age_vector[1] +
                       self.data.contact_data[country]["HOME"][15, 2] * age_vector[2])) / age_2
            # third column
            # age group 0-4, 15-19
            col_0_2 = self.data.contact_data[country]["HOME"][0, 3]
            # age group 5-14, 15-19
            col_1_2 = (self.data.contact_data[country]["HOME"][1, 3] * age_vector[3] +
                       self.data.contact_data[country]["HOME"][2, 3] * age_vector[3]) / age_2
            # age group 15-19, 15-19
            col_2_2 = self.data.contact_data[country]["HOME"][3, 3]
            # age group 20-24, 15-19
            col_3_2 = self.data.contact_data[country]["HOME"][4, 3]
            col_4_2 = (self.data.contact_data[country]["HOME"][5, 3] * age_vector[5] +
                       self.data.contact_data[country]["HOME"][6, 3] * age_vector[6] +
                       self.data.contact_data[country]["HOME"][7, 3] * age_vector[7] +
                       self.data.contact_data[country]["HOME"][8, 3] * age_vector[8] +
                       self.data.contact_data[country]["HOME"][9, 3] * age_vector[9] +
                       self.data.contact_data[country]["HOME"][10, 3] * age_vector[10] +
                       self.data.contact_data[country]["HOME"][11, 3] * age_vector[11] +
                       self.data.contact_data[country]["HOME"][12, 3] * age_vector[12]) / (age_5)

            col_5_2 = (self.data.contact_data[country]["HOME"][13, 3] * age_vector[13] +
                       self.data.contact_data[country]["HOME"][14, 3] * age_vector[14] +
                       self.data.contact_data[country]["HOME"][15, 3] * age_vector[15]) / (age_6)

            # 4th column
            # age group 0-4, 20-24
            col_0_3 = self.data.contact_data[country]["HOME"][0, 4]
            # age group 5-14, 20-24
            col_1_3 = (self.data.contact_data[country]["HOME"][1, 4] * age_vector[4] +
                       self.data.contact_data[country]["HOME"][2, 4] * age_vector[4]) / age_2
            # age group 15-19, 20-24
            col_2_3 = self.data.contact_data[country]["HOME"][3, 4]
            # age group 20-24, 15-19
            col_3_3 = self.data.contact_data[country]["HOME"][4, 4]
            col_4_3 = (self.data.contact_data[country]["HOME"][5, 4] * age_vector[5] +
                       self.data.contact_data[country]["HOME"][6, 4] * age_vector[6] +
                       self.data.contact_data[country]["HOME"][7, 4] * age_vector[7] +
                       self.data.contact_data[country]["HOME"][8, 4] * age_vector[8] +
                       self.data.contact_data[country]["HOME"][9, 4] * age_vector[9] +
                       self.data.contact_data[country]["HOME"][10, 4] * age_vector[10] +
                       self.data.contact_data[country]["HOME"][11, 4] * age_vector[11] +
                       self.data.contact_data[country]["HOME"][12, 4] * age_vector[12]) / age_5

            col_5_3 = (self.data.contact_data[country]["HOME"][13, 4] * age_vector[13] +
                       self.data.contact_data[country]["HOME"][14, 4] * age_vector[14] +
                       self.data.contact_data[country]["HOME"][15, 4] * age_vector[15]) / age_6

            # 5th column
            col_0_4 = (self.data.contact_data[country]["HOME"][0, 5] * age_vector[5] +
                       self.data.contact_data[country]["HOME"][0, 6] * age_vector[6] +
                       self.data.contact_data[country]["HOME"][0, 7] * age_vector[7] +
                       self.data.contact_data[country]["HOME"][0, 8] * age_vector[8] +
                       self.data.contact_data[country]["HOME"][0, 9] * age_vector[9] +
                       self.data.contact_data[country]["HOME"][0, 10] * age_vector[10] +
                       self.data.contact_data[country]["HOME"][0, 11] * age_vector[11] +
                       self.data.contact_data[country]["HOME"][0, 12] * age_vector[12]) / (age_5)

            col_1_4 = (self.data.contact_data[country]["HOME"][1, 5] * age_vector[5] +
                       self.data.contact_data[country]["HOME"][1, 6] * age_vector[6] +
                       self.data.contact_data[country]["HOME"][1, 7] * age_vector[7] +
                       self.data.contact_data[country]["HOME"][1, 8] * age_vector[8] +
                       self.data.contact_data[country]["HOME"][1, 9] * age_vector[9] +
                       self.data.contact_data[country]["HOME"][1, 10] * age_vector[10] +
                       self.data.contact_data[country]["HOME"][1, 11] * age_vector[11] +
                       self.data.contact_data[country]["HOME"][1, 12] * age_vector[12] +
                       self.data.contact_data[country]["HOME"][2, 5] * age_vector[5] +
                       self.data.contact_data[country]["HOME"][2, 6] * age_vector[6] +
                       self.data.contact_data[country]["HOME"][2, 7] * age_vector[7] +
                       self.data.contact_data[country]["HOME"][2, 8] * age_vector[8] +
                       self.data.contact_data[country]["HOME"][2, 9] * age_vector[9] +
                       self.data.contact_data[country]["HOME"][2, 10] * age_vector[10] +
                       self.data.contact_data[country]["HOME"][2, 11] * age_vector[11] +
                       self.data.contact_data[country]["HOME"][2, 12] * age_vector[12]) / (age_5)

            col_2_4 = (self.data.contact_data[country]["HOME"][3, 5] * age_vector[5] +
                       self.data.contact_data[country]["HOME"][3, 6] * age_vector[6] +
                       self.data.contact_data[country]["HOME"][3, 7] * age_vector[7] +
                       self.data.contact_data[country]["HOME"][3, 8] * age_vector[8] +
                       self.data.contact_data[country]["HOME"][3, 9] * age_vector[9] +
                       self.data.contact_data[country]["HOME"][3, 10] * age_vector[10] +
                       self.data.contact_data[country]["HOME"][3, 11] * age_vector[11] +
                       self.data.contact_data[country]["HOME"][3, 12] * age_vector[12]) / (age_5)

            col_3_4 = (self.data.contact_data[country]["HOME"][4, 5] * age_vector[5] +
                       self.data.contact_data[country]["HOME"][4, 6] * age_vector[6] +
                       self.data.contact_data[country]["HOME"][4, 7] * age_vector[7] +
                       self.data.contact_data[country]["HOME"][4, 8] * age_vector[8] +
                       self.data.contact_data[country]["HOME"][4, 9] * age_vector[9] +
                       self.data.contact_data[country]["HOME"][4, 10] * age_vector[10] +
                       self.data.contact_data[country]["HOME"][4, 11] * age_vector[11] +
                       self.data.contact_data[country]["HOME"][4, 12] * age_vector[12]) / (age_5)

            col_4_4 = ((self.data.contact_data[country]["HOME"][5, 5] + self.data.contact_data[country]["HOME"][6, 5] +
                        self.data.contact_data[country]["HOME"][7, 5] + self.data.contact_data[country]["HOME"][8, 5] +
                        self.data.contact_data[country]["HOME"][9, 5] + self.data.contact_data[country]["HOME"][10, 5] +
                        self.data.contact_data[country]["HOME"][11, 5] +
                        self.data.contact_data[country]["HOME"][12, 5]) * age_vector[5] +
                       (self.data.contact_data[country]["HOME"][5, 6] + self.data.contact_data[country]["HOME"][6, 6] +
                        self.data.contact_data[country]["HOME"][7, 6] + self.data.contact_data[country]["HOME"][8, 6] +
                        self.data.contact_data[country]["HOME"][9, 6] + self.data.contact_data[country]["HOME"][10, 6] +
                        self.data.contact_data[country]["HOME"][11, 6] +
                        self.data.contact_data[country]["HOME"][12, 6]) * age_vector[6] +
                       (self.data.contact_data[country]["HOME"][5, 7] + self.data.contact_data[country]["HOME"][6, 7] +
                        self.data.contact_data[country]["HOME"][7, 7] + self.data.contact_data[country]["HOME"][8, 7] +
                        self.data.contact_data[country]["HOME"][9, 7] + self.data.contact_data[country]["HOME"][10, 7] +
                        self.data.contact_data[country]["HOME"][11, 7] +
                        self.data.contact_data[country]["HOME"][12, 7]) * age_vector[7] +
                       (self.data.contact_data[country]["HOME"][5, 8] + self.data.contact_data[country]["HOME"][6, 8] +
                        self.data.contact_data[country]["HOME"][7, 8] + self.data.contact_data[country]["HOME"][8, 8] +
                        self.data.contact_data[country]["HOME"][9, 8] + self.data.contact_data[country]["HOME"][10, 8] +
                        self.data.contact_data[country]["HOME"][11, 8] +
                        self.data.contact_data[country]["HOME"][12, 8]) * age_vector[8] +
                       (self.data.contact_data[country]["HOME"][5, 9] + self.data.contact_data[country]["HOME"][6, 9] +
                        self.data.contact_data[country]["HOME"][7, 9] + self.data.contact_data[country]["HOME"][8, 9] +
                        self.data.contact_data[country]["HOME"][9, 9] + self.data.contact_data[country]["HOME"][10, 9] +
                        self.data.contact_data[country]["HOME"][11, 9] +
                        self.data.contact_data[country]["HOME"][12, 9]) * age_vector[9] +
                       (self.data.contact_data[country]["HOME"][10, 5] + self.data.contact_data[country]["HOME"][
                           10, 6] +
                        self.data.contact_data[country]["HOME"][10, 7] + self.data.contact_data[country]["HOME"][
                            10, 8] +
                        self.data.contact_data[country]["HOME"][10, 9] + self.data.contact_data[country]["HOME"][
                            10, 10] +
                        self.data.contact_data[country]["HOME"][11, 10] +
                        self.data.contact_data[country]["HOME"][12, 10]) * age_vector[10] +
                       (self.data.contact_data[country]["HOME"][11, 5] + self.data.contact_data[country]["HOME"][
                           11, 6] +
                        self.data.contact_data[country]["HOME"][11, 7] + self.data.contact_data[country]["HOME"][
                            11, 8] +
                        self.data.contact_data[country]["HOME"][11, 9] + self.data.contact_data[country]["HOME"][
                            11, 10] +
                        self.data.contact_data[country]["HOME"][11, 11] +
                        self.data.contact_data[country]["HOME"][11, 12]) * age_vector[11] +
                       (self.data.contact_data[country]["HOME"][12, 5] + self.data.contact_data[country]["HOME"][
                           12, 6] +
                        self.data.contact_data[country]["HOME"][12, 7] + self.data.contact_data[country]["HOME"][
                            12, 8] +
                        self.data.contact_data[country]["HOME"][12, 9] + self.data.contact_data[country]["HOME"][
                            12, 10] +
                        self.data.contact_data[country]["HOME"][12, 11] +
                        self.data.contact_data[country]["HOME"][12, 12]) * age_vector[12]) / age_5

            col_5_4 = ((self.data.contact_data[country]["HOME"][13, 5] + self.data.contact_data[country]["HOME"][
                13, 6] +
                        self.data.contact_data[country]["HOME"][13, 7] + self.data.contact_data[country]["HOME"][
                            13, 8] +
                        self.data.contact_data[country]["HOME"][13, 9] + self.data.contact_data[country]["HOME"][
                            13, 10] +
                        self.data.contact_data[country]["HOME"][13, 11] +
                        self.data.contact_data[country]["HOME"][13, 12]) * age_vector[13] +
                       (self.data.contact_data[country]["HOME"][14, 5] + self.data.contact_data[country]["HOME"][
                           14, 6] +
                        self.data.contact_data[country]["HOME"][14, 7] + self.data.contact_data[country]["HOME"][
                            14, 8] +
                        self.data.contact_data[country]["HOME"][14, 9] + self.data.contact_data[country]["HOME"][
                            14, 10] +
                        self.data.contact_data[country]["HOME"][14, 11] +
                        self.data.contact_data[country]["HOME"][14, 12]) * age_vector[14] +
                       (self.data.contact_data[country]["HOME"][15, 5] + self.data.contact_data[country]["HOME"][
                           15, 6] +
                        self.data.contact_data[country]["HOME"][15, 7] + self.data.contact_data[country]["HOME"][
                            15, 8] +
                        self.data.contact_data[country]["HOME"][15, 9] + self.data.contact_data[country]["HOME"][
                            15, 10] +
                        self.data.contact_data[country]["HOME"][15, 11] +
                        self.data.contact_data[country]["HOME"][15, 12]) * age_vector[12]) / age_5
            # 6th column
            col_0_5 = (self.data.contact_data[country]["HOME"][0, 13] * age_vector[13] +
                       self.data.contact_data[country]["HOME"][0, 14] * age_vector[14] +
                       self.data.contact_data[country]["HOME"][0, 15] * age_vector[15]) / age_6
            col_1_5 = ((self.data.contact_data[country]["HOME"][1, 13] +
                        self.data.contact_data[country]["HOME"][2, 13]) * age_vector[13] +
                       (self.data.contact_data[country]["HOME"][1, 14] +
                        self.data.contact_data[country]["HOME"][2, 14]) * age_vector[14] +
                       (self.data.contact_data[country]["HOME"][1, 15] +
                        self.data.contact_data[country]["HOME"][2, 15]) * age_vector[15]) / age_6
            col_2_5 = (self.data.contact_data[country]["HOME"][3, 13] * age_vector[13] +
                       self.data.contact_data[country]["HOME"][3, 14] * age_vector[14] +
                       self.data.contact_data[country]["HOME"][3, 15] * age_vector[15]) / age_6
            col_3_5 = (self.data.contact_data[country]["HOME"][4, 13] * age_vector[13] +
                       self.data.contact_data[country]["HOME"][4, 14] * age_vector[14] +
                       self.data.contact_data[country]["HOME"][4, 15] * age_vector[15]) / age_6
            col_4_5 = ((self.data.contact_data[country]["HOME"][5, 13] +
                        self.data.contact_data[country]["HOME"][6, 13] +
                        self.data.contact_data[country]["HOME"][7, 13] +
                        self.data.contact_data[country]["HOME"][8, 13] +
                        self.data.contact_data[country]["HOME"][9, 13] +
                        self.data.contact_data[country]["HOME"][10, 13] +
                        self.data.contact_data[country]["HOME"][11, 13] +
                        self.data.contact_data[country]["HOME"][12, 13]) * age_vector[13] +
                       (self.data.contact_data[country]["HOME"][5, 14] +
                        self.data.contact_data[country]["HOME"][6, 14] +
                        self.data.contact_data[country]["HOME"][7, 14] +
                        self.data.contact_data[country]["HOME"][8, 14] +
                        self.data.contact_data[country]["HOME"][9, 14] +
                        self.data.contact_data[country]["HOME"][10, 14] +
                        self.data.contact_data[country]["HOME"][11, 14] +
                        self.data.contact_data[country]["HOME"][12, 14]) * age_vector[14] +
                       (self.data.contact_data[country]["HOME"][5, 15] +
                        self.data.contact_data[country]["HOME"][6, 15] +
                        self.data.contact_data[country]["HOME"][7, 15] +
                        self.data.contact_data[country]["HOME"][8, 15] +
                        self.data.contact_data[country]["HOME"][9, 15] +
                        self.data.contact_data[country]["HOME"][10, 15] +
                        self.data.contact_data[country]["HOME"][11, 15] +
                        self.data.contact_data[country]["HOME"][12, 15]) * age_vector[15]) / age_6

            col_5_5 = ((self.data.contact_data[country]["HOME"][13, 13] +
                        self.data.contact_data[country]["HOME"][14, 13] +
                        self.data.contact_data[country]["HOME"][15, 13]) * age_vector[13] +
                       (self.data.contact_data[country]["HOME"][13, 14] +
                        self.data.contact_data[country]["HOME"][14, 14] +
                        self.data.contact_data[country]["HOME"][15, 14]) * age_vector[14] +
                       (self.data.contact_data[country]["HOME"][13, 15] +
                        self.data.contact_data[country]["HOME"][14, 15] +
                        self.data.contact_data[country]["HOME"][15, 15]) * age_vector[15]) / age_6

            cont_home = np.zeros(shape=(6, 6))
            cont_home[0, 0] = col_0_0
            cont_home[1, 0] = col_1_0
            cont_home[2, 0] = col_2_0
            cont_home[3, 0] = col_3_0
            cont_home[4, 0] = col_4_0
            cont_home[5, 0] = col_5_0
            cont_home[0, 1] = col_0_1
            cont_home[1, 1] = col_1_1
            cont_home[2, 1] = col_2_1
            cont_home[3, 1] = col_3_1
            cont_home[4, 1] = col_4_1
            cont_home[5, 1] = col_5_1
            cont_home[0, 2] = col_0_2
            cont_home[1, 2] = col_1_2
            cont_home[2, 2] = col_2_2
            cont_home[3, 2] = col_3_2
            cont_home[4, 2] = col_4_2
            cont_home[5, 2] = col_5_2
            cont_home[0, 3] = col_0_3
            cont_home[1, 3] = col_1_3
            cont_home[2, 3] = col_2_3
            cont_home[3, 3] = col_3_3
            cont_home[4, 3] = col_4_3
            cont_home[5, 3] = col_5_3
            cont_home[0, 4] = col_0_4
            cont_home[1, 4] = col_1_4
            cont_home[2, 4] = col_2_4
            cont_home[3, 4] = col_3_4
            cont_home[4, 4] = col_4_4
            cont_home[5, 4] = col_5_4
            cont_home[0, 5] = col_0_5
            cont_home[1, 5] = col_1_5
            cont_home[2, 5] = col_2_5
            cont_home[3, 5] = col_3_5
            cont_home[4, 5] = col_4_5
            cont_home[5, 5] = col_5_5
            self.contact_home.update(
                {country: cont_home}
            )


def main():
    data_tr = DataTransformer()
    ind = Home(data_tr=data_tr, country_names=data_tr.country_names)
    ind.get_contact_home()


if __name__ == "__main__":
    main()
