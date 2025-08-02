import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Metricas para a avaliação do modelo
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

plt.style.use('seaborn-v0_8')
class AvaliacaoMD:
    def __init__(self):
        pass

    def diferencaModeloFinalBL(self, df):
        ax = sns.barplot(data=df, x="Metrica", y="Valores", hue="Modelo")

        # Adicionar os valores em cima das barras
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=10)
            
        plt.title("Modelo Final vs Baseline")
        plt.xlabel("Metrica")
        plt.ylabel('Valor')
        plt.show()

    def importanciaVariaveis(self, X_train, melhorModelo):
        #Criando uma base de dados com os valores de influencia por coluna
        colunasQueInfluenciamNaDecisão = {
            "Colunas" : X_train.columns,
            "Influencia" : melhorModelo.feature_importances_
        }
        colunasQueInfluenciamNaDecisão = pd.DataFrame(colunasQueInfluenciamNaDecisão)
        ordenadoPorColunasQueInfluenciamNaDecisão = colunasQueInfluenciamNaDecisão.sort_values(by = "Influencia", ascending=False)


        influencias = ordenadoPorColunasQueInfluenciamNaDecisão["Influencia"] / ordenadoPorColunasQueInfluenciamNaDecisão["Influencia"].max()

        # Criar um gradiente de cores (quanto maior a importância, mais escura a barra)
        colors = plt.cm.Blues(influencias)  # Usando o colormap 'Blues'

        # Plotar o gráfico de barras com cores baseadas na importância
        plt.figure(figsize=(10, 6))
        bars = plt.bar(ordenadoPorColunasQueInfluenciamNaDecisão["Colunas"], ordenadoPorColunasQueInfluenciamNaDecisão["Influencia"] , color=colors)
        plt.xticks(range(len(ordenadoPorColunasQueInfluenciamNaDecisão["Colunas"])), ordenadoPorColunasQueInfluenciamNaDecisão["Colunas"].values, rotation=90)
        plt.xlabel('Variáveis')
        plt.ylabel('Importância')
        plt.title('Importância das Variáveis no Modelo (Cores por Influência)')
    
    def avaliacaoDeEscalaOriginal(self, y_true_log, y_pred_log):
        # Conversão para escala original
        y_true_orig = np.exp(y_true_log)
        y_pred_orig = np.exp(y_pred_log)
        
        # Cálculo das métricas
        metrics = {
            'MAE_R$': mean_absolute_error(y_true_orig, y_pred_orig),
            'MAE_%': mean_absolute_percentage_error(y_true_orig, y_pred_orig) * 100,
            'R2_%': r2_score(y_true_orig, y_pred_orig) * 100,
            'RMSE_R$': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
            'Custo_Médio_R$': np.mean(y_true_orig)
        }
        
        # Cálculo adicional de comparação
        metrics['RMSE_%'] = (metrics['RMSE_R$'] / metrics['Custo_Médio_R$']) * 100
        
        # Formatação de impressão profissional
        print("\n" + "="*60)
        print("AVALIAÇÃO DO MODELO NA ESCALA ORIGINAL (R$)".center(60))
        print("="*60)
        print(f"• MAE (Erro Absoluto Médio):       R$ {metrics['MAE_R$']:,.2f} ({metrics['MAE_%']:.2f}%)".replace(",", "X").replace(".", ",").replace("X", "."))
        print(f"• RMSE (Raiz do Erro Quadrático):  R$ {metrics['RMSE_R$']:,.2f} ({metrics['RMSE_%']:.2f}%)".replace(",", "X").replace(".", ",").replace("X", "."))
        print(f"• R² (Poder Explicativo):          {metrics['R2_%']:.2f}%")
        print(f"• Custo Médio de Referência:       R$ {metrics['Custo_Médio_R$']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        print("="*60)