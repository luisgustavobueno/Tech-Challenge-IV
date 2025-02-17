# Tech Challenge <04>

## Grupo 2

- Julio Cesario de Paiva Leão (julio0023@live.com)
- Luis Gustavo Bueno Colombo (luisgustavobuenocolombo@gmail.com)

> Antes de tudo, veja vídeo gerado após o experimento :wink:

[![Vídeo gerado](/img/ezgif-6a3a803d42cd87.gif)](/output_video.mp4)

## Objetivo

Nesse projeto temos o objetivo de analisar uma vídeo e implementar técnicas de reconhecimento facial e detecção de movimentos e emoções, através de algoritmos python de Deep Learning.

Em resumo, esse projeto visa atender os seguintes requisitos:

1. **Reconhecimento facial**: Identificar e marcar os rostos presentes no vídeo.
2. **Análise de expressões emocionais**: Analisar as expressões emocionais dos rostos identificados.
3. **Detecção de atividades**: Detectar e categorizar atividades como mão levantada e movimento da cabeça.
4. **Detecção de anomalias**: Identificar movimentos bruscos ou comportamentos atípicos.
5. **Geração de resumo**: Criar um resumo automático das principais atividades, emoções e anomalias detectadas no vídeo.

## URLs do projeto

- [Vídeo do YouTube](https://youtu.be/Upi7jtRrp_g)
- [Repositório do GitHub](https://github.com/luisgustavobueno/Tech-Challenge-IV)

## Como Instalar e Executar

### Pré-requisitos

- [Python 3.10+](https://www.python.org/downloads/) instalado.
- `cv2`: Para manipulação de vídeos e imagens.
- `mediapipe`: Para detecção de poses e movimentos humanos.
- `face_recognition`: Para reconhecimento facial.
- `deepface`: Para análise de emoções faciais.
- `os`: Para manipulação de arquivos e diretórios.
- `tqdm`: Para exibir uma barra de progresso durante o processamento.

### Ambiente virtual

Crie e ative o ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### Instalação das Dependências

Execute o seguinte comando para instalar as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

### Inciando a aplicação

1. Coloque o vídeo que deseja analisar na mesma pasta do script Python.
2. Modifique a variável `input_video_path` no código para o nome do seu arquivo de vídeo.
3. Execute o script Python:

```bash
python face_emotion_detection.py
```

> A execução total leva um determinado tempo a depender do poder de processamento da CPU.
>
> Por ex. Um Ryzen 5 5600G leva em torno de 35 minutos.

O vídeo processado será salvo como `output_video.mp4`, e o resumo será exibido no terminal.

## Estrutura do código

### Variáveis globais

#### `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`

Configura o uso da CPU em vez de utilizar a GPU para o processamento, o que pode ser útil em máquinas sem suporte a CUDA.

#### `SHOW_VIDEO = False`

Define se o vídeo processado será exibido ou não. Modifique para `True` para visualizar o vídeo enquanto ele é processado.

#### `FACE_INFO = {}`

Dicionário utilizado para armazenar informações sobre os rostos detectados e suas emoções.

#### `FRAME_SKIP = 1`

Define a quantidade de frames a serem pulados entre cada processamento. Modifique para um valor maior se desejar aumentar a velocidade de processamento, mas isso pode diminuir a precisão.

#### `EMOTION_THRESHOLD = 95`

Define o limiar de sensibilidade para a detecção de emoções faciais. Valores maiores resultam em maior sensibilidade.

#### `ANOMALY_THRESHOLD = 0.2`

Define o limiar de sensibilidade para a detecção de anomalias nos movimentos de pose.

#### `HEAD_MOVEMENT_THRESHOLD = 0.1`

Define o limiar de sensibilidade para a detecção de movimentos na cabeça.

#### `last_dominant_emotion = None`

Armazena a última emoção dominante detectada.

### Classe `PoseDetector`

Responsável por detectar poses e movimentos no vídeo.

- `__init__`: Inicializa o detector de poses.
- `process_frame`: Processa um frame para detectar landmarks corporais.
- `draw_landmarks`: Desenha os landmarks detectados no frame.
- `is_hand_up`: Verifica se a mão está levantada.
- `is_head_moving`: Verifica se a cabeça está se movendo.
- `detect_movements`: Detecta movimentos específicos (mão levantada e movimento da cabeça).

### Classe `FaceRecognition`

Responsável por reconhecer rostos e analisar expressões emocionais.

- `__init__`: Inicializa o reconhecedor de rostos e carrega rostos conhecidos.
- `load_known_faces`: Carrega rostos conhecidos de um diretório.
- `recognize_faces`: Reconhece rostos em um frame.
- `detect_emotion`: Analisa a emoção dominante em um rosto detectado.

### Funções Auxiliares

- `display_info`: Exibe informações sobre os rostos detectados no frame, incluindo nome e emoção detectada.
- `format_faces_output`: Formata a saída de rostos detectados, armazenando as emoções associadas aos rostos no dicionário `FACE_INFO`.
- `display_movement_text`: Exibe na tela informações sobre os movimentos detectados (como mãos levantadas e cabeça se movendo).
- `detect_anomalies`: Detecta anomalias em movimentos humanos, verificando diferenças significativas nas posições da cabeça entre frames consecutivos.
- `generate_summary`: Gera um resumo detalhado da análise do vídeo, incluindo o número de frames analisados, anomalias detectadas e os rostos reconhecidos.
- `save_summary_to_file`: Salva o resumo em um arquivo `txt`.
- `process_video`: Função principal que processa o vídeo, realizando a detecção de rostos, emoções e movimentos. Também gera o resumo e salva o vídeo processado.

## Execução

A função `process_video` recebe dois parâmetros:

- `input_path`: Caminho do vídeo a ser analisado.
- `output_path`: Caminho do vídeo processado que será salvo.

O código processa o vídeo frame por frame, detectando rostos, movimentos e emoções. Além disso, ele gera e salva um resumo sobre os rostos reconhecidos, emoções detectadas e movimentos observados.

## Exemplo de Uso

Para rodar o código em seu ambiente, basta colocar o caminho do vídeo de entrada e o caminho do vídeo de saída no final do código:

```python
if __name__ == "__main__":
    input_video_path = "seu_video.mp4"
    output_video_path = "video_processado.mp4"
    process_video(input_video_path, output_video_path)
```

Isso irá processar o vídeo `seu_video.mp4` e salvar o vídeo processado em `video_processado.mp4`, além de gerar um arquivo de resumo chamado `summary.txt`.

## Saída do Relatório

Ao final da execução, o código gera um resumo no terminal e um arquivo `summary.txt` com as seguintes informações:

1. **Total de frames analisados**: Número total de frames processados no vídeo.
2. **Número de anomalias detectadas**: Quantidade de movimentos bruscos ou comportamentos atípicos identificados.
3. **Movimentos detectados**: Contagem de movimentos específicos (mão levantada e movimento da cabeça).
4. **Rostos reconhecidos**: Lista de rostos identificados e suas emoções dominantes.

### Resumo de saída

```bash
Total de frames analisados: 3326
Número de anomalias detectadas: 10
Movimentos detectados: {'hand_up': 44, 'head_moving': 19}
Rostos reconhecidos:
{
    'Desconhecido': ['happy', 'surprise', 'neutral', 'sad'],
    'Joao': ['happy', 'neutral', 'sad'],
    'Ana': ['happy', 'sad', 'neutral', 'surprise', 'fear', 'angry'],
    'Bruno': ['neutral', 'happy'],
    'Maria': ['happy', 'surprise', 'neutral', 'angry', 'fear'],
    'Jhon': ['neutral', 'happy', 'angry', 'sad', 'surprise']
}
```

## Personalização

- **Sensibilidade de detecção de emoções**: Você pode ajustar o valor de `EMOTION_THRESHOLD` para modificar a sensibilidade.
- **Sensibilidade de detecção de anomalias**: Ajuste o valor de `ANOMALY_THRESHOLD` para controlar a detecção de anomalias.
- **Sensibilidade de movimentos da cabeça**: Ajuste o valor de `HEAD_MOVEMENT_THRESHOLD` para maior ou menor sensibilidade ao movimento da cabeça.

## Considerações Finais

Este código oferece uma análise detalhada de vídeos com foco em rostos, emoções e movimentos humanos, sendo uma base para aplicações de vigilância, interação e análise comportamental.
