# 📊 Previsão de Custos de Seguros Médicos com Machine Learning

## 📝 Descrição do Projeto

Este projeto tem como objetivo **prever os custos de seguros médicos de pacientes** com base em características pessoais e comportamentais. Através da análise exploratória de dados, seleção de variáveis e aplicação de algoritmos de aprendizado de máquina, buscamos identificar os fatores que mais influenciam os gastos com seguros e construir um modelo robusto para realizar previsões confiáveis.

---

## 🔍 Principais Insights

- As variáveis com **maior impacto nos custos** dos seguros são:
  - **Fumante (`smoker_yes`)**
  - **Idade (`age`)**
- Fumantes apresentam **custos médios significativamente mais altos**.
- O modelo teve **ganhos importantes de desempenho** após o ajuste de hiperparâmetros.
- Métricas como o **Erro Médio Absoluto (MAE)** e o **R² Score** indicaram boa capacidade preditiva do modelo final.

---

## 📌 Etapas do Projeto

### 1. 📥 Importação e Visualização Inicial dos Dados
- Carregamento do dataset com informações como idade, sexo, índice de massa corporal (BMI), número de filhos, tabagismo, região e custos médicos.
- Análise exploratória básica para entender a distribuição e correlação entre as variáveis.

### 2. 🧼 Tratamento dos Dados
- Verificação e tratamento de valores nulos.
- Codificação de variáveis categóricas (ex: `sex`, `smoker`, `region`) com **One-Hot Encoding**.
- Padronização de colunas numéricas quando necessário.

### 3. 📈 Análise Exploratória de Dados (EDA)
- Visualização de relações entre variáveis (gráficos de dispersão, boxplots, heatmaps de correlação).
- Identificação de outliers e padrões importantes.
- Compreensão dos **grupos mais impactados** pelos custos.

### 4. 🧠 Avaliação das Features
- Aplicação de técnicas como **Feature Importance** para verificar quais variáveis mais contribuem para o modelo.
- Seleção das variáveis mais relevantes para a previsão.

### 5. 🤖 Treinamento e Avaliação de Modelos
- Teste com diferentes algoritmos:
  - Regressão Linear
  - Árvores de Decisão
  - Random Forest
  - Gradient Boosting
- Avaliação com métricas como:
  - **MAE (Erro Médio Absoluto)**
  - **MSE (Erro Quadrático Médio)**
  - **R² Score**

### 6. 🛠️ Ajuste de Hiperparâmetros
- Uso de técnicas como **GridSearchCV** ou **RandomizedSearchCV**.
- Comparação do desempenho **antes e depois** do ajuste.
- Seleção do melhor modelo com base no equilíbrio entre **precisão e generalização**.

### 7. 📊 Análise Final de Importância das Variáveis
- Geração de gráfico de importância das variáveis.
- Interpretação dos resultados com destaque para as variáveis mais relevantes nos custos.

---

## ✅ Conclusão

O modelo final demonstrou boa capacidade de previsão e permitiu **entender com clareza os principais fatores que impactam os custos com seguros médicos**. As informações geradas podem ser usadas por seguradoras para definir políticas de preços mais justas e por instituições de saúde para desenvolver ações de prevenção mais direcionadas.
