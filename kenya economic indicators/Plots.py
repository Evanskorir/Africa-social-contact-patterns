
from Indicators import Indicators
from Dataloader import DataLoader


class Plots:
    def __init__(self, pca_data):
        self.pca_data = pca_data
        self.data = DataLoader()

    def run(self):
        analysis = Indicators()
        analysis.PCA_apply()
        analysis.project_2D()


def main():

    # execute class indicators
    # Create plots for the paper
    ind = Indicators()
    ind.corr_data()
    ind.PCA_apply()
    # ind.loadings()
    ind.corr_pcs()
    ind.dendogram_pca()
    ind.components_project()
    # ind.project_2D()
    ind.plot_counties()
    ind.box_plot()


if __name__ == "__main__":
    main()
