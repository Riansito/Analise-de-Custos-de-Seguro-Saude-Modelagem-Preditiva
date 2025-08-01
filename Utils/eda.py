import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class AnaliseEDA:
    def __init__(self):
        pass


    def analiseUnivariada(self, df, tipo):
        colunas = df.select_dtypes(include = tipo).columns
        if tipo == "number":
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize = (10, 5))
            axs=axs.flatten()
            for i, coluna in enumerate(colunas):
                axs[i].hist(df[coluna])
                axs[i].set_xlabel(coluna)
            plt.suptitle("Distribuição das variaveis numericas")
        else:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (15, 5))
            axs=axs.flatten()

            for i, coluna in enumerate(colunas):
                agrupamento = df[coluna].value_counts().reset_index()
                agrupamento.columns = [coluna, "Quantidade"]
                axs[i].bar(agrupamento[coluna], agrupamento["Quantidade"])
                for j, valor in enumerate(agrupamento["Quantidade"]):
                    axs[i].text(j, valor, str(valor), fontsize=12, ha='center', va='bottom')
                axs[i].set_xlabel(coluna)
            plt.suptitle("Distribuição das variaveis categoricas")
        plt.tight_layout()
        plt.show()