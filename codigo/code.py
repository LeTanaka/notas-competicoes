# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:05:22 2021

@author: lueij
"""
# importando bibliotecas pandas e numpy
import pandas as pd
import numpy as np

# lendo planilha excel
df_projeto = pd.read_excel('apresentacao_projeto.xlsx',header=1)
df_nacionais = pd.read_excel('competicoes_nacionais.xlsx',header=1)
df_regionais = pd.read_excel('competicoes_regionais.xlsx',header=1)
df_nacional2k16 = pd.read_excel('nacional_2016.xlsx',header=1,skipfooter=0,sheet_name="classificacao_final")
df_nacional2k16 = df_nacional2k16.rename(columns=({"Classificação":"Posicao",
                                                   "Número":"Num",
                                                   "Nome":"Nome_equipes",
                                                   "Total de Pontos":"Total_pontos",
                                                   "Penalizações":"Penalizacoes",
                                                   "Segurança":"Seguranca",
                                                   "Conforto":"Conforto",
                                                   "Relatório":"Relatorio",
                                                   "Apresentação":"Apresentacao",
                                                   "Finais de Projeto":"Finais_projeto",
                                                   "Aceleração":"Aceleracao",
                                                   "Velocidade":"Velocidade",
                                                   "Tração":"Tracao",
                                                   "S&T":"S&T",
                                                   "Old School":"Old_school",
                                                   "Enduro":"Enduro"
                                                   }
                                                  ))
# dados irrelevante
df_nacional2k16 = df_nacional2k16.drop(['Num','Old_school','Finais_projeto'], axis=1)

# bibliotecas para o algoritmo de regressao, dividir dataset entre treino e teste, plotar graficos
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# x = classificacao na competicao
x = df_nacional2k16.drop(['Posicao','Nome_equipes','Penalizacoes','Seguranca'], axis=1)
y = df_nacional2k16['Posicao']
x = x.rename(columns=({"Total de Pontos":"Total_pontos",
                       "Conforto":"Conforto",
                       "Relatório":"Relatorio",
                       "Apresentação":"Apresentacao",
                       "Aceleração":"Aceleracao",
                       "Velocidade":"Velocidade",
                       "Tração":"Tracao",
                       "S&T":"S&T",
                       "Enduro":"Enduro"
                       }
                       ))
# colocando dados de teste no modelo
#x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
ml = LinearRegression()
#ml.fit(x_train, y_train)

#y_pred = ml.predict(x_test)
#previsao = ml.predict([[833.32, 18.8, 129.3, 112.97, 50.69, 48.12, 44.38, 13.6, 377.78]])

# para calcular precisao do modelo (2016) criado
from sklearn.metrics import r2_score

#precisao = r2_score(y_test, y_pred)
#print(precisao)


df_nacional2k14 = pd.read_excel('nacional_2014.xlsx',header=1)

y2 = df_nacional2k14['Posição Competição']

y2 = pd.concat([y,y2])
df_nacional2k14 = df_nacional2k14.drop(['Nº Carro', 'Equipe', 'Universidade', 'Segurança', 'MudBog 20 anos', 'Posição Competição'], axis=1)

df_nacional2k14 = df_nacional2k14.rename(columns=({"Pontos Competição":"Total_pontos",
                                                   "Conforto":"Conforto",
                                                   "Relatório":"Relatorio",
                                                   "Apresentação":"Apresentacao",
                                                   "Aceleração":"Aceleracao",
                                                   "Velocidade":"Velocidade",
                                                   "Tração":"Tracao",
                                                   "S& T":"S&T",
                                                   "Enduro":"Enduro"
                                                   }))

df_nacional2k14 = df_nacional2k14[['Total_pontos','Conforto','Relatorio','Apresentacao', 'Aceleracao', 'Velocidade', 'Tracao', 'S&T', 'Enduro']]
x2 = pd.concat([x, df_nacional2k14])
x2 = x2.rename(columns=({"0":"Total_pontos",
                                                   "1":"Conforto",
                                                   "2":"Relatorio",
                                                   "3":"Apresentacao",
                                                   "4":"Aceleracao",
                                                   "5":"Velocidade",
                                                   "6":"Tracao",
                                                   "7":"S&T",
                                                   "8":"Enduro"
                                                   }))

# treinando modelo com mais dados (2016,2015)
x_train, x_test, y_train, y_test = train_test_split(x2, y2,test_size = 0.3, random_state=0)
ml.fit(x_train,y_train)

# prova velocidade = manobrabilidade (2019)
y_pred = ml.predict(x_test)
prev_geral = ml.predict([[822.44, 18.80, 129.30, 112.97, 50.69, 48.18, 44.38, 13.60, 377.78]])
#print(y_pred)
print(abs(prev_geral))
precisao = r2_score(y_test, y_pred)
print(precisao)

# plotar grafico
plt.figure(figsize=(15,10))
plt.scatter(y_test, y_pred)
plt.xlabel('Real')
plt.ylabel('Previsto')
plt.title('Real vs Previsto')

