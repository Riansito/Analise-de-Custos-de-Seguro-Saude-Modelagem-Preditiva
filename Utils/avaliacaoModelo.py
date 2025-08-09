#Manipulação e graficos com os dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#Pipeline de tratamento dos dados
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline

#Dividir os dados em treino e teste
from sklearn.model_selection import train_test_split

#Metricas para a avaliação do modelo
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

plt.style.use('seaborn-v0_8')
class AvaliacaoMD:
    def __init__(self):
        pass

    def binaria_para_num(self, X):
        #Recebe os dados do X(somentes as colunas binarias) deixa os valores em 0 e 1
        X = X.copy()
        for col in X.columns:
            if (col == "smoker"):
                X[col] = X[col].map({'yes': 1, 'no': 0})
            else:
                X[col] = X[col].map({'male': 1, 'female': 0})
        return X

    def tratamentoTreinoTeste(self, X_train, X_test):
        colunas_numericas = ["age", "bmi", "children"]
        colunas_binarias = ["smoker", "sex"]
        colunas_categoricas = ["region"]

        bin_transformer = Pipeline(steps=[
            ('bin_to_num', FunctionTransformer(self.binaria_para_num))
        ])

        #Tranforma os dados com as metodologias certas para cada coluna(As colunas numerics não tiveram nenhum tipo de tratamento)
        processador_colunas = ColumnTransformer(
            transformers=[
                ('bin', bin_transformer, colunas_binarias),
                ('num', 'passthrough', colunas_numericas),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), colunas_categoricas)
            ]
        )

        processador_colunas.fit(X_train)
        X_train_transformado = processador_colunas.transform(X_train)
        X_test_transformado = processador_colunas.transform(X_test)


        # Nomes das colunas após transformação
        ohe = processador_colunas.named_transformers_['cat']
        nomes_ohe = ohe.get_feature_names_out(colunas_categoricas)

        colunas_final = colunas_binarias + colunas_numericas + list(nomes_ohe) 

        # DataFrames prontos
        X_train_df = pd.DataFrame(X_train_transformado, columns=colunas_final, index=X_train.index)
        X_test_df = pd.DataFrame(X_test_transformado, columns=colunas_final, index=X_test.index)

        return X_train_df, X_test_df

    def avaliaModelo(self, y_true, y_pred, X):
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred),
            'Adjusted R²': 1 - (1-r2_score(y_true, y_pred))*(len(y_true)-1)/(len(y_true)-X.shape[1]-1)
        }
        return metrics

    def testaModelo(self, X, y, modelo):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = modelo()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        metricas = self.avaliaModelo(y_test, y_pred, X)
        print(modelo)
        for name, value in metricas.items():
            print(f"{name}: {value:.4f}")


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
        y_true_orig = np.expm1(y_true_log)
        y_pred_orig = np.expm1(y_pred_log)
        
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