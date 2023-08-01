# Projeto realizado em sala de aula
# Aplicação de Modelos de Machine Learning
# Curso: Aplicações Informáticas para Ciência de Dados
# Professor: Tiago Cunha Reis, PhD
# Ipluso - Instituto Politécnico da lusofonia
# Autora: Bianca Lima

#01 Bibliotecas necessárias
#! pip install matplotlib scikit-learn
#! pip install -r requirements.txt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#02 Gerando dados fictícios (idade, pressao_arterial)
idade = np.random.randint(20, 80, size=(100, 1))
pressao_arterial = 100 + idade * 0.7 + np.random.normal(scale=5, size=(100, 1))

#03 Representação gráfica dos dados gerados
fig, axs = plt.subplots(nrows=1, ncols=2)

titles = 'Idade', 'Pressão Arterial'
for idx, v in enumerate([idade, pressao_arterial]):
    axs[idx].hist(v, color='#6A5ACD', edgecolor='k')
    axs[idx].set_title(titles[idx])

plt.savefig('Histograms')

#04 Criar um modelo Regressão Linear
model = LinearRegression()
model.fit(idade, pressao_arterial)

#04.1 Previsão
idades_teste = np.linspace(start=20, stop=80, num=5).reshape(-1, 1)
pred = model.predict(idades_teste)

for idx, el in enumerate(idades_teste):
    print(f'Para {el[0]} anos, previsão = {pred[idx][0]:.2f} mmHg')

#05 Representação gráfica
plt.figure()
plt.scatter(idade, pressao_arterial)
plt.plot(idades_teste, pred, color='red', label='RL')

plt.xlabel('Idade (anos)')
plt.ylabel('Pressão Arterial (mmHg)')

plt.legend()
plt.savefig('Scatter.png')
