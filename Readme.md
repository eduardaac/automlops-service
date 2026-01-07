# AutoMLOps - Arquitetura Auto-Adaptativa para Machine Learning
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112.2-009688?style=for-the-badge&logo=fastapi)
![Grafana](https://img.shields.io/badge/Grafana-Latest-F46800?style=for-the-badge&logo=grafana&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Latest-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)

## 📋 Índice
- [Visão Geral](#-visão-geral)
- [Arquitetura](#-arquitetura)
- [Tecnologias](#-tecnologias)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Monitoramento](#-monitoramento)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [API Endpoints](#-api-endpoints)
- [Contribuição](#-contribuição)

---

## 🎯 Visão Geral

**AutoMLOps** é uma plataforma completa para operacionalização de Machine Learning com recursos de:

- ✅ **AutoML**: Seleção automática de modelos com PyCaret
- 📊 **Monitoramento**: Dashboard em tempo real com Grafana + Prometheus
- 🔄 **Champion/Challenger**: Sistema de comparação de modelos
- 🚨 **Alertas**: Detecção automática de degradação de performance
- 🏥 **Health Index**: Métrica unificada combinando Drift (50%) + Confidence (30%) + Anomaly (20%)
- 🐳 **Docker**: Deploy simplificado com Docker Compose

### 🎨 Features Principais

| Feature | Descrição |
|---------|-----------|
| **Treinamento Automatizado** | PyCaret AutoML com 15+ algoritmos |
| **Sistema Champion/Challenger** | Comparação automática entre modelos |
| **Monitoramento Visual** | Dashboards Grafana|
| **Health Index (Métrica Real)** | Proxy unificado: Drift (50%) + Confidence (30%) + Anomaly (20%) |
| **Data Drift Detection** | Alertas de mudança na distribuição |
| **MLflow Tracking** | Rastreamento de experimentos |
| **API REST** | FastAPI com documentação automática |

---

## 🛠️ Tecnologias

### Core Stack
| Tecnologia | Versão | Função |
|------------|--------|--------|
| **Python** | 3.11.8 | Linguagem principal |
| **Poetry** | Latest | Gerenciamento de dependências |
| **FastAPI** | 0.112.2 | Framework web |
| **Uvicorn** | Latest | ASGI server |

### Machine Learning
| Tecnologia | Versão | Função |
|------------|--------|--------|
| **PyCaret** | 3.3.1 | AutoML framework |
| **Scikit-learn** | 1.5.1 | Algoritmos ML |
| **LightGBM** | 4.4.0 | Gradient boosting |
| **XGBoost** | 2.1.0 | Gradient boosting |
| **CatBoost** | 1.2.5 | Gradient boosting |
| **MLflow** | 2.10.0 | Experiment tracking |

### Monitoramento
| Tecnologia | Versão | Função |
|------------|--------|--------|
| **Grafana** | Latest | Visualização de métricas |
| **Prometheus** | Latest | Coleta de métricas |
| **prometheus-client** | 0.20.0 | Python SDK |

### Database
| Tecnologia | Versão | Função |
|------------|--------|--------|
| **PostgreSQL** | 13 | Banco principal |
| **SQLAlchemy** | 2.0.31 | ORM |
| **Alembic** | 1.13.2 | Migrações |

### Infraestrutura
| Tecnologia | Versão | Função |
|------------|--------|--------|
| **Docker** | Latest | Containerização |
| **Docker Compose** | Latest | Orquestração |

---

## 💻 Pré-requisitos

### Requisitos Mínimos
- **Sistema Operacional**: Windows 10/11, Linux, macOS
- **Python**: 3.11 ou superior
- **RAM**: 4GB (8GB recomendado)
- **Disco**: 2GB livres
- **Docker**: 20.10+ (opcional)
- **Poetry**: 1.5+

### Instalação de Dependências

#### Windows
```powershell
# Instalar Python 3.11
winget install Python.Python.3.11

# Instalar Poetry
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Instalar Docker Desktop
winget install Docker.DockerDesktop
```

#### Linux/macOS
```bash
# Instalar Python 3.11
sudo apt-get install python3.11 python3.11-venv  # Ubuntu/Debian
brew install python@3.11                          # macOS

# Instalar Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Instalar Docker
curl -fsSL https://get.docker.com | sh
```

---

## 🚀 Instalação

### Passo 1: Clonar o Repositório

```bash
git clone https://github.com/eduardaac/TCC.git
cd TCC/automlops_api
```

### Passo 2: Configurar Ambiente Python

```bash
# Instalar dependências com Poetry
poetry install

# Ativar ambiente virtual
poetry shell
```

### Passo 3: Configurar Variáveis de Ambiente

```bash
# Criar arquivo .env (opcional - valores padrão já estão configurados)
# Edite apenas se precisar customizar:

# DATABASE_URL=postgresql://automlops_user:sua_senha@127.0.0.1:5432/automlops_db
# MLFLOW_TRACKING_URI=http://127.0.0.1:5000
# PORT=8000
# IP=127.0.0.1
```

**Nota**: O projeto já vem com configurações padrão funcionais. Só crie o `.env` se precisar customizar.

### Passo 4: Iniciar Serviços Docker

```bash
# Iniciar todos os serviços
docker-compose up -d

# Verificar status
docker ps
```

**Serviços iniciados:**
- ✅ Grafana (http://localhost:3000)
- ✅ Prometheus (http://localhost:9090)
- ✅ PostgreSQL (localhost:5432)
- ✅ MLflow (http://localhost:5000)

### Passo 5: Iniciar API

```bash
# Windows PowerShell
poetry run python main.py

# Linux/macOS
poetry run python main.py
```

**Verificar se porta está livre (Windows)**:
```powershell
# Se porta 8000 estiver em uso
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
```

**API estará disponível em:**
- 🌐 API: http://localhost:8000
- 📚 Documentação: http://localhost:8000/docs
- 🔧 Redoc: http://localhost:8000/redoc

---

## 📁 Estrutura do Projeto

```
automlops_api/
├── 📄 main.py                      # Entry point da aplicação
├── 📄 pyproject.toml               # Dependências Poetry
├── 📄 docker-compose.yml           # Orquestração de containers
├── 📄 Dockerfile                   # Imagem Docker da API
├── 📄 prometheus.yml               # Configuração Prometheus
├── 📄 .env.example                 # Template de variáveis
├── 📄 Readme.md                    # Este arquivo
│
├── 📂 src/                         # Código-fonte principal
│   ├── 📄 app.py                   # Configuração FastAPI
│   │
│   ├── 📂 routers/                 # Endpoints da API
│   │   ├── training.py            # Rotas de treinamento
│   │   ├── prediction.py          # Rotas de predição
│   │   ├── models.py              # Rotas de modelos
│   │   ├── monitoring.py          # Rotas de métricas
│   │   └── human_actions.py       # Rotas de intervenção
│   │
│   ├── 📂 services/                # Lógica de negócio
│   │   ├── training_service.py    # Serviço de treinamento
│   │   ├── prediction_service.py  # Serviço de predição
│   │   ├── model_service.py       # Gerenciamento de modelos
│   │   ├── performance_service.py # Avaliação de performance
│   │   ├── alert_service.py       # Sistema de alertas
│   │   └── file_service.py        # Manipulação de arquivos
│   │
│   ├── 📂 middleware/              # Middlewares
│   │   ├── metrics_middleware.py  # Coleta de métricas
│   │   └── metrics_sync.py        # Sincronização DB→Prometheus
│   │
│   ├── 📂 database/                # Persistência
│   │   ├── config.py              # Configuração SQLAlchemy
│   │   └── models/                # Modelos de dados
│   │       ├── File.py            # Tabela files
│   │       ├── Result.py          # Tabela results
│   │       ├── Alert.py           # Tabela alerts
│   │       └── PerformanceLog.py  # Tabela performance_logs
│   │
│   ├── 📂 schemas/                 # Modelos Pydantic
│   │   ├── training.py            # DTOs de treinamento
│   │   ├── prediction.py          # DTOs de predição
│   │   ├── monitoring.py          # DTOs de monitoramento
│   │   └── common.py              # DTOs compartilhados
│   │
│   ├── 📂 classes/                 # Classes auxiliares
│   │   ├── AutoML.py              # Wrapper PyCaret
│   │   └── Model.py               # Abstração de modelo
│   │
│   └── 📂 utils/                   # Utilitários
│       ├── automl_handler.py      # Manipulação AutoML
│       ├── data_validator.py      # Validação de dados
│       ├── check_data_drift.py    # Detecção de drift
│       ├── converter.py           # Conversões de dados
│       ├── file_utils.py          # Utilitários de arquivo
│       └── monitoring_observer.py # Observador de métricas
│
├── 📂 grafana/                     # Configuração Grafana
│   ├── 📂 dashboards/
│   │   ├── automlops-dashboards.json  # Dashboard principal
│   │   └── dashboards.yml         # Provisionamento
│   └── 📂 datasources/
│       └── datasources.yml        # Datasource Prometheus
│
├── 📂 mlruns/                      # MLflow tracking (gitignored)
├── 📂 mlartifacts/                 # MLflow artifacts (gitignored)
├── 📂 tmp/                         # Arquivos temporários (gitignored)
│   ├── files/                     # Uploads temporários
│   ├── models/                    # Modelos serializados
│   └── prediction_results/        # Resultados de predição
│
└── 📂 logs/                        # Logs da aplicação (gitignored)
```

---

## 👥 Autores

- **Eduarda** - [@eduardaac](https://github.com/eduardaac)

---

## 📧 Contato

- **GitHub**: [@eduardaac](https://github.com/eduardaac)
- **Repository**: [TCC](https://github.com/eduardaac/TCC)

---

## 📚 Referências

- [PyCaret Documentation](https://pycaret.gitbook.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

