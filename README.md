# Snake-brazuca
Código do jogo clássico snake com inteligência artificial, mais especificamente usando aprendizado por reforço, o objetivo é demonstrar conceitos básicos de aprendizado por reforço em um jogo divertido :) 

Regras:
Comeu a laranja? Ganha ponto
Bateu na parede? Perde ponto
Andou pela jogo? Ganha ponto

A rede neural transforma a execução dessas regras em calibração da rede neural, e a cada atualização da rede com a tabela q ("q values") o agente escolhe a melhor ação.

Entendendo o aprendizado por reforço

Agente =
Estado = 
Ação = 

Os arquivos principais são:

game.py: este arquivo contém o código para o jogo Snake em si. A classe Jogo define as funções para inicializar o jogo, atualizar o estado do jogo e verificar se o jogo terminou.

agente.py: este arquivo contém o código para o agente que joga o jogo. O agente utiliza um modelo de rede neural para tomar decisões com base no estado atual do jogo. A classe Agente define as funções para treinar o modelo e para tomar decisões de ação durante o jogo.

model.py: este arquivo contém o código para a definição do modelo de rede neural utilizado pelo agente. A classe Modelo define a arquitetura da rede neural e as funções para calcular os valores Q para cada estado de jogo.

helper.py: este arquivo contém funções auxiliares para o treinamento do modelo e para a exibição do jogo. A função get_state é utilizada para obter o estado atual do jogo, a função to_tensor é utilizada para converter os dados para tensores PyTorch, e a função play_game é utilizada para exibir o jogo com o agente treinado.

Para executar o jogo com o agente treinado, basta executar o arquivo agente.py. Este arquivo carrega o modelo treinado a partir do arquivo model.pth, exibe o jogo com o agente treinado e permite que o usuário jogue o jogo manualmente.

Para executar o arquivo, abra o terminal na pasta raiz do projeto e execute o comando:

# Jogando  
python agente.py
Ao jogar o jogo, o agente irá aprender a maximizar a pontuação por meio do algoritmo de Q-learning. O modelo de rede neural utilizado pelo agente foi treinado previamente utilizando o arquivo train.py, que salva o modelo treinado no arquivo model.pth.

Para treinar o modelo, basta executar o arquivo agente.py. Este arquivo treina o modelo utilizando o algoritmo de Q-learning e salva o modelo treinado no arquivo model.pth na pasta model.

Durante o treinamento, é exibido um gráfico com as pontuações do agente em cada jogo, bem como a média das pontuações dos últimos 100 jogos.

Os requisitos para executar o código são: Python 3, PyTorch, Matplotlib e NumPy. As bibliotecas podem ser instaladas utilizando o gerenciador de pacotes pip. Para instalar as bibliotecas, execute os seguintes comandos no terminal:

pip install torch
pip install matplotlib
pip install numpy

Obs:O snake não está passando de 70 pontos, em breve farei atualizações no código para otimizar seus parâmetros.
