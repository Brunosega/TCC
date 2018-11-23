# Avaliação de técnicas de aprendizagem profunda na verificação de faces

Esse GitHub é destinado à entrega da documentação de instalação e execução dos testes do projeto.

## Compatibilidade
A execução dos scripts foi realizada utilizando Tensorflow 1.7, Python 3.6 em Ubuntu simulado por Anaconda.

## Credibilidade
Os scripts de treinamento e testes foram implementados por [David Sanberg](https://github.com/davidsandberg/facenet) e adaptados por Bruno Cattalini.

## Bases de Treinamento e Testes
Foram utilizadas as bases [AT&T](), [FERET]() e [YALE]().

## Executando o treinamento

### Instalar o tensorflow
É necessária a utilização do Tensorflow 1.7, onde em seu [guia](https://www.tensorflow.org/install/#pip_installation) explica como realizar a instalação da CUDA e cuDNN em sua GPU (necessário).

### Clonar esse repositório
Usando o comando:
```
git clone https://github.com/Brunosega/TCC.git
```

### Iniciando o treinamento
Os parâmetros indicados abaixo representam os utilizados durante o projeto, apenas alterando os referentes às bases de dados de treinamento (data_dir) e local de salvamento do modelo (models_base_dir) e log(logs_base_dir).

Para o treinamento a partir do método Softmax, execute o comando: 
```
python facenet/src/train_softmax.py 
--logs_base_dir "/results/TCC/logs/facenet/triplet/att/" 
--models_base_dir "/models/facenet/softmax/att/" 
--data_dir "/datasets/Originais/att_faces_160/" 
--image_size 160 
--model_def models.inception_resnet_v1 
--optimizer ADAM 
--learning_rate 0.01 
--max_nrof_epochs 10  
--weight_decay 5e-4 
--embedding_size 512 
--batch_size 30 
--lfw_dir "/datasets/lfw/lfw_mtcnnpy_160" 
--lfw_pairs "/facenet/data/pairs.txt"
```

Para o treinamento a partir do método Tripletloss, execute o comando com os parâmetros: 
```
python facenet/src/train_tripletloss.py 
--logs_base_dir "/results/TCC/logs/facenet/triplet/att/" 
--models_base_dir "/models/facenet/triplet/att/" 
--data_dir "/datasets/Originais/att_faces_160/" 
--image_size 160 
--model_def models.inception_resnet_v1 
--optimizer ADAGRAD 
--learning_rate 0.01 
--max_nrof_epochs 10  
--weight_decay 5e-4 
--embedding_size 512 
--batch_size 30 
--people_per_batch 30 
--images_per_person 6 
--lfw_dir "/datasets/lfw/lfw_mtcnnpy_160" 
--lfw_pairs "/facenet/data/pairs.txt"
```
## Modelos Treinados
Os modelos treinados estão separados nas pastas /models/triplet e /models/softmax.

## Resultados
Após a execução dos treinamentos e testes, os resultados estão separados nas pastas /results/testes/triplet e /results/testes/softmax.
