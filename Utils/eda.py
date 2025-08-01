import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8')
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
    
   
    def analiseBivariada(self, df, tipo):
        colunas = df.select_dtypes(include = tipo).columns
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize = (15, 5))
        axs=axs.flatten()
        if tipo == "number":
            for i, coluna in enumerate(colunas[:3]):
                axs[i].scatter(df[coluna], df["charges"])
                axs[i].set_xlabel(coluna)
                axs[i].set_ylabel("Charges")
            plt.suptitle("Relação entre as variaveis numericas em relação a target")
        else:
            for i, coluna in enumerate(colunas):
                agrupamento = df[[coluna, "charges"]].groupby(coluna).mean().sort_values(by = "charges", ascending = False).reset_index()
                agrupamento.columns = [coluna, "Média"]
                axs[i].bar(agrupamento[coluna], agrupamento["Média"])
                for j, valor in enumerate(agrupamento["Média"]):
                    mediaFormatada = f"R$ {valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                    axs[i].text(j, valor, mediaFormatada, fontsize=12, ha='center', va='bottom')
                axs[i].set_xlabel(coluna)
            plt.suptitle("Relação entre as variaveis categóricas em relação a target")
        plt.tight_layout()
        plt.show()


    def analiseMultivariada(self, df, tipo, hue = "smoker"):
        
        colunas = df.select_dtypes(include=tipo).columns
        
        if tipo == "number":
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            axs = axs.flatten()
            for i, coluna in enumerate(colunas[:3]):
                sns.scatterplot(data=df, x=coluna, y="charges", hue=df[hue], ax=axs[i])
                axs[i].set_xlabel(coluna)
                axs[i].set_ylabel("Charges")
            plt.suptitle(f"Variáveis numéricas vs Charges com hue = '{hue}'", fontsize=14)

        else:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
            axs = axs.flatten()
            for i, coluna in enumerate(["sex", "region"]):
                sns.violinplot(data=df, x=coluna, y="charges", hue=hue, split=True, ax=axs[i])
                axs[i].set_ylabel("Custos")
                axs[i].set_xlabel(coluna)
                axs[i].tick_params(axis='x', rotation=30)
            plt.suptitle(f"Variáveis categóricas vs Charges com hue = '{hue}'", fontsize=14)

        plt.tight_layout()
        plt.show()