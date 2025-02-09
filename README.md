# Documentação do Código de Análise de Vídeo com Reconhecimento Facial e Detecção de Atividades

## Objetivo do Código

O objetivo deste código é realizar a análise de vídeos para:

1. **Reconhecimento facial**: Identificar e marcar os rostos presentes no vídeo.
2. **Análise de expressões emocionais**: Analisar as expressões emocionais dos rostos identificados.
3. **Detecção de atividades**: Detectar e categorizar atividades como mão levantada e movimento da cabeça.
4. **Detecção de anomalias**: Identificar movimentos bruscos ou comportamentos atípicos.
5. **Geração de resumo**: Criar um resumo automático das principais atividades, emoções e anomalias detectadas no vídeo.

## Como Instalar e Executar

### Pré-requisitos

- [Python 3.10+](https://www.python.org/downloads/) instalado.

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

### Execução do Código

1. Coloque o vídeo que deseja analisar na mesma pasta do script Python.
2. Modifique a variável `input_video_path` no código para o nome do seu arquivo de vídeo.
3. Execute o script Python:

```bash
python face_emotion_detection.py
```

O vídeo processado será salvo como `output_video.mp4`, e o resumo será exibido no terminal.

## Explicando as funções

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

- `display_info`: Exibe informações sobre rostos reconhecidos e emoções no frame.
- `format_faces_output`: Formata as informações de nomes e emoções dos rostos encontrados para o relatório final.
- `display_movement_text`: Exibe contagens de movimentos detectados no frame.
- `detect_anomalies`: Detecta movimentos bruscos ou comportamentos atípicos.
- `generate_summary`: Gera um resumo das análises realizadas no vídeo.
- `save_summary_to_file`: Salva o resumo em um arquivo `txt`.
- `process_video`: Processa o vídeo frame a frame, aplicando todas as funcionalidades.

## Saída do Relatório

Ao final da execução, o código gera um resumo no terminal e um arquivo `summary.txt` com as seguintes informações:

1. **Total de frames analisados**: Número total de frames processados no vídeo.
2. **Número de anomalias detectadas**: Quantidade de movimentos bruscos ou comportamentos atípicos identificados.
3. **Movimentos detectados**: Contagem de movimentos específicos (mão levantada e movimento da cabeça).
4. **Rostos reconhecidos**: Lista de rostos identificados e suas emoções dominantes.

### Exemplo de Saída:

```plaintext
Resumo do Vídeo:
Total de frames analisados: 1500
Número de anomalias detectadas: 12
Movimentos detectados: {'hand_up': 5, 'head_moving': 8}
Rostos reconhecidos: {'João': ['happy','angry'], 'Maria': ['neutral'], 'Desconhecido':, ['sad']}
```

## Considerações Finais

Este código é uma solução completa para análise de vídeos, combinando reconhecimento facial, detecção de emoções, monitoramento de atividades e identificação de anomalias. Ele pode ser adaptado para diferentes cenários, como monitoramento de segurança, análise de comportamento ou estudos de interação humana.
