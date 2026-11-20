# Detecção de Fraudes Financeiras com Machine Learning

Protótipo funcional desenvolvido para a disciplina de **Mineração de Dados (SIS920)** — Curso de Sistemas de Informação (Unex, Vitória da Conquista - BA).

O sistema resolve o trade-off operacional entre prejuízos causados por *chargebacks* (fraudes não detectadas) e a perda de receita por falsos positivos (clientes legítimos bloqueados indevidamente). A solução utiliza modelos de classificação supervisionada treinados sobre mais de 284 mil transações reais, entregando predições em milissegundos e simulação de impacto financeiro em uma interface interativa via Streamlit.

---

## Funcionalidades da Aplicação

A interface (`app.py`) está estruturada em três módulos principais para apoio à tomada de decisão:

1. **Painel do Gestor:** Comparativo técnico entre **Random Forest** e **Regressão Logística**, exibindo matrizes de confusão e métricas reais de desempenho (evitando a ilusão da acurácia bruta em bases desbalanceadas).
2. **Simulador em Tempo Real:** Avaliação individual de transações inéditas (conjunto de teste), retornando a probabilidade de fraude e classificando a operação em três faixas configuráveis na barra lateral (**Aprovar**, **Revisar Manualmente** ou **Bloquear**).
3. **Simulação Financeira:** Projeção de ganho ou prejuízo líquido mensal (baseada em 1 milhão de transações/mês) com premissas ajustáveis em reais para o custo de fraude e atrito com clientes bloqueados.

---

## Arquitetura do Projeto

```text
├── data/
│   └── creditcard.csv          # Base de dados bruta (não versionada no Git)
├── models/                     # Artefatos dos modelos treinados gerados pelo pipeline
├── src/
│   └── features.py             # Funções de preparação (preparar_dados) e balanceamento (balancear_treino)
├── mineracao.py                # Script principal de ETL, treinamento e serialização dos modelos
├── app.py                      # Aplicação web interativa (Streamlit)
├── requirements.txt            # Dependências do ambiente Python
└── README.md                   # Documentação e guia de reprodução

```

### Pipeline de Dados e Modelagem

* **Engenharia de Atributos (`src/features.py`):** Conversão do tempo acumulado (`Time`) na variável cíclica `Hour` ($Hour = (Time / 3600) \pmod{24}$) e padronização de escalas com `StandardScaler` nos atributos `Amount` e `Hour` para alinhamento com as componentes PCA (`V1`–`V28`).
* **Balanceamento de Classes:** Aplicação da técnica **SMOTE** (*Synthetic Minority Over-sampling Technique*) exclusivamente sobre o conjunto de treino para corrigir a proporção original de ~0,2% de fraudes.
* **Desempenho Comparativo:** O modelo **Random Forest** destacou-se por identificar ~86% das fraudes (84 de 98 no conjunto de validação) gerando apenas 65 falsos positivos em 56 mil transações, enquanto a Regressão Logística apresentou alto índice de bloqueios indevidos (1.534 clientes legítimos).

---

## Guia de Reprodução e Execução

Siga os passos abaixo para configurar e executar o projeto do zero em um ambiente limpo.

### 1. Pré-requisitos

* Python 3.9 ou superior instalado.
* Conta no Kaggle (ou ferramenta CLI configurada) para obtenção da base de dados pública.

### 2. Clonar o Repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO

```

### 3. Criar Ambiente Virtual e Instalar Dependências

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual (Windows)
venv\Scripts\activate

# Ativar ambiente virtual (Linux/macOS)
source venv/bin/activate

# Instalar bibliotecas necessárias
pip install -r requirements.txt

```

### 4. Download e Preparação da Base de Dados

1. Acesse o dataset oficial [Credit Card Fraud Detection no Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) e faça o download do arquivo `creditcard.csv` (ou utilize a Kaggle CLI):
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip

```


2. Certifique-se de que o arquivo esteja salvo exatamente no caminho:
```text
data/creditcard.csv

```



### 5. Treinar e Gerar os Modelos

Execute o pipeline de mineração para processar os dados, aplicar o SMOTE e salvar os modelos treinados:

```bash
python mineracao.py --dados data/creditcard.csv

```

*(Aguarde a conclusão do treinamento antes de iniciar a interface).*

### 6. Iniciar a Aplicação Web

Suba o servidor local do Streamlit para acessar os painéis e o simulador:

```bash
python -m streamlit run app.py

```

O navegador abrirá automaticamente no endereço `http://localhost:8501`.

---

## Testes Unitários e Validação

Para validar as rotinas de pré-processamento e balanceamento implementadas em `src/features.py` (`preparar_dados` e `balancear_treino`), execute a suíte de testes:

```bash
pytest

```

---

## Autores

* **Danilo Motta Rodrigues**
* **Gabriel da Silva Carvalho**
* **Pedro Santos Simões da Silva**
* **Vitor Hugo Lima Silva Bittencourt**

**Orientador:** Prof. Me. Wallas Froes de Oliveira — Unex (2026)
