# Outlier Detection Application

Aplicação de microsserviços para detecção de outliers em séries temporais de preços de ações usando LSTM Autoencoder com Atenção Multi-Head.

## Arquitetura

```
outlier_detection_application/
├── artifacts/                    # Modelos e artefatos salvos
│   ├── forecast_lstm_ae.pth     # Pesos do modelo treinado
│   ├── scaler.gz                # Scaler para normalização
│   └── model_config.json        # Configuração do modelo
│
├── src/
│   ├── shared/                  # Código compartilhado
│   │   ├── __init__.py
│   │   ├── model.py            # Arquitetura do modelo LSTM
│   │   ├── config.py           # Configurações
│   │   └── data_processing.py  # Processamento de dados
│   │
│   ├── training/               # Serviço de treinamento
│   │   ├── __init__.py
│   │   └── train.py           # Script de treinamento
│   │
│   └── prediction/             # Serviço de predição (FastAPI)
│       ├── __init__.py
│       ├── api.py             # API FastAPI
│       └── schemas.py         # Schemas Pydantic
│
└── requirements.txt
```

## Instalação

```bash
cd outlier_detection_application
pip install -r requirements.txt
```

## Uso

### 1. Treinar o Modelo

Execute o serviço de treinamento para criar os artefatos do modelo:

```bash
cd src/training
python train.py
```

Isso irá:
- Baixar dados históricos de ações (AAPL por padrão)
- Treinar o modelo LSTM Autoencoder
- Salvar os artefatos em `artifacts/`

### 2. Iniciar o Serviço de Predição

Após o treinamento, inicie a API FastAPI:

```bash
cd src/prediction
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Acessar a Documentação da API

Abra no navegador:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Endpoints da API

### Health Check
```
GET /health
```
Verifica o status do serviço.

### Informações do Modelo
```
GET /model/info
```
Retorna informações sobre a arquitetura do modelo.

### Predição Única
```
POST /predict
```
Faz predição para uma única sequência de entrada.

**Exemplo de Request:**
```json
{
  "sequence": {
    "values": [
      [0.01, 17.5, 0.02],
      [0.02, 17.8, 0.03],
      ...
    ]
  }
}
```

### Predição em Lote
```
POST /predict/batch
```
Faz predições para múltiplas sequências.

### Detecção de Outliers
```
POST /detect-outliers
```
Analisa dados de ações e detecta anomalias.

**Exemplo de Request:**
```json
{
  "ticker": "AAPL",
  "start_date": "2020-01-01",
  "end_date": "2025-12-01",
  "sigma_multiplier": 3.0
}
```

## Modelo

### Arquitetura

O modelo utiliza:
- **Encoder LSTM**: Processa sequências temporais
- **Atenção Multi-Head**: Captura dependências importantes
- **Decoder**: Reconstrói/prediz o próximo timestep

### Features de Entrada

1. **Return**: Variação percentual do preço de fechamento
2. **LogVolume**: Log do volume de negociação
3. **HighLowSpread**: Spread normalizado entre máxima e mínima

### Detecção de Anomalias

Anomalias são detectadas baseando-se no erro de reconstrução:
- Erro > threshold = Anomalia
- Threshold calculado usando distribuição log-normal (média + 3σ)

## Configuração

Edite `src/shared/config.py` para ajustar:
- Hiperparâmetros do modelo
- Período de dados
- Caminhos de artefatos
