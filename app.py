import json
import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Detector de Fraudes', layout='wide')


@st.cache_resource
def carregar_modelo():
    return joblib.load('models/melhor_modelo.joblib')


@st.cache_data
def carregar_metricas():
    with open('reports/metricas.json', encoding='utf-8') as arquivo:
        return json.load(arquivo)


@st.cache_data
def carregar_teste():
    if os.path.exists('data/creditcard.csv'):
        df = pd.read_csv('data/creditcard.csv')
        completa = True
    else:
        df = pd.read_csv('data/creditcard_amostra.csv')
        completa = False

    df['Hour'] = (df['Time'] / 3600) % 24
    x = df.drop(columns=['Class', 'Time'])
    y = df['Class']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    teste = x_test.copy()
    teste['Class'] = y_test
    return teste, completa


def formatar_reais(valor):
    texto = 'R$ {:,.0f}'.format(valor)
    return texto.replace(',', '.')


def calcular_risco(valores, pacote):
    tabela = pd.DataFrame([valores])
    tabela[['Amount', 'Hour']] = pacote['scaler'].transform(tabela[['Amount', 'Hour']])
    tabela = tabela[pacote['colunas']]
    return float(pacote['modelo'].predict_proba(tabela)[0][1])


if not os.path.exists('models/melhor_modelo.joblib') or not os.path.exists('reports/metricas.json'):
    st.error('Modelo nao encontrado. Rode primeiro: python mineracao.py --dados data/creditcard.csv')
    st.stop()

pacote = carregar_modelo()
metricas = carregar_metricas()
base, base_completa = carregar_teste()
colunas_v = [c for c in pacote['colunas'] if c.startswith('V')]


def sortear(classe):
    filtro = base[base['Class'] == classe]
    if len(filtro) == 0:
        return False
    linha = filtro.sample(1).iloc[0]
    valores = {}
    for coluna in colunas_v:
        valores[coluna] = float(linha[coluna])
    valores['Amount'] = float(linha['Amount'])
    valores['Hour'] = float(linha['Hour'])
    st.session_state['transacao'] = valores
    st.session_state['classe_real'] = int(linha['Class'])
    return True


if 'transacao' not in st.session_state:
    sortear(0)

st.title('Detector de Fraudes em Cartao de Credito')
st.caption('Modelo em uso: ' + pacote['nome'])

st.sidebar.header('Regras de decisao')
limiar_revisao = st.sidebar.slider('Enviar para revisao a partir de', 0.05, 0.95, 0.30, 0.05)
limiar_bloqueio = st.sidebar.slider('Bloquear a partir de', 0.05, 0.95, 0.70, 0.05)
if limiar_revisao > limiar_bloqueio:
    st.sidebar.warning('O limite de revisao deve ser menor ou igual ao de bloqueio.')

aba1, aba2, aba3 = st.tabs(['Simulador de transacao', 'Painel do gestor', 'Simulacao financeira'])

with aba1:
    st.subheader('Analise de uma transacao')
    st.write('As variaveis V1 a V28 sao anonimizadas pela base original, por isso a transacao e sorteada do conjunto de teste, que o modelo nunca viu no treino. Valor e hora podem ser alterados.')
    if not base_completa:
        st.warning('creditcard.csv nao encontrado: usando a amostra. As transacoes sorteadas podem ter sido vistas no treino.')

    col_a, col_b = st.columns(2)
    if col_a.button('Sortear transacao legitima'):
        sortear(0)
    if col_b.button('Sortear transacao fraudulenta'):
        if not sortear(1):
            st.warning('Nao ha fraudes nesta base. Use o creditcard.csv completo.')

    transacao = st.session_state['transacao']

    col_c, col_d = st.columns(2)
    valor = col_c.number_input('Valor da compra', min_value=0.0, value=float(transacao['Amount']))
    horas_padrao = int(transacao['Hour'])
    minutos_padrao = int((transacao['Hour'] - horas_padrao) * 60)
    col_d1, col_d2 = col_d.columns(2)
    hora_inteira = col_d1.number_input('Hora (0 a 23)', min_value=0, max_value=23, value=horas_padrao, step=1)
    minuto = col_d2.number_input('Minuto (0 a 59)', min_value=0, max_value=59, value=minutos_padrao, step=1)
    hora = hora_inteira + minuto / 60

    valores_finais = dict(transacao)
    valores_finais['Amount'] = valor
    valores_finais['Hour'] = hora

    risco = calcular_risco(valores_finais, pacote)

    st.metric('Probabilidade de fraude', str(round(risco * 100, 1)) + '%')
    st.progress(min(max(risco, 0.0), 1.0))

    if risco >= limiar_bloqueio:
        st.error('DECISAO: BLOQUEAR a transacao')
    elif risco >= limiar_revisao:
        st.warning('DECISAO: ENVIAR PARA REVISAO MANUAL')
    else:
        st.success('DECISAO: APROVAR a transacao')

    if st.session_state['classe_real'] == 1:
        st.caption('Na base original, esta transacao e uma fraude.')
    else:
        st.caption('Na base original, esta transacao e legitima.')

with aba2:
    st.subheader('Desempenho dos modelos no conjunto de teste')
    linhas = []
    for nome, m in metricas['metricas'].items():
        linhas.append({
            'Modelo': nome,
            'Precisao': round(m['precisao'], 3),
            'Recall': round(m['recall'], 3),
            'F1': round(m['f1'], 3),
            'PR-AUC': round(m['pr_auc'], 3),
            'ROC-AUC': round(m['roc_auc'], 3),
            'Acuracia': round(m['acuracia'], 4),
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True)
    st.caption('A acuracia fica perto de 100% ate para um modelo ruim, porque so cerca de 0,17% das transacoes sao fraude. Por isso a metrica principal e a PR-AUC.')

    melhor = metricas['melhor_modelo']
    mc = metricas['metricas'][melhor]['matriz_confusao']
    st.write('Matriz de confusao do melhor modelo: ' + melhor)
    matriz = pd.DataFrame(
        [[mc['tn'], mc['fp']], [mc['fn'], mc['tp']]],
        index=['Real: legitima', 'Real: fraude'],
        columns=['Previsto: legitima', 'Previsto: fraude'],
    )
    st.dataframe(matriz)

    col_e, col_f = st.columns(2)
    if os.path.exists('reports/curva_precisao_recall.png'):
        col_e.image('reports/curva_precisao_recall.png')
    if os.path.exists('reports/curva_roc.png'):
        col_f.image('reports/curva_roc.png')

with aba3:
    st.subheader('Quanto a solucao economiza?')
    st.write('Ajuste os valores abaixo para a realidade da empresa. Os resultados sao estimativas baseadas no desempenho medido no teste.')

    col_g, col_h = st.columns(2)
    volume = col_g.number_input('Transacoes por mes', min_value=1000, value=1000000, step=100000)
    taxa = col_h.number_input('Taxa de fraude (%)', min_value=0.01, value=0.17, step=0.01)

    col_i, col_j = st.columns(2)
    ticket = col_i.number_input('Prejuizo medio por fraude (R$)', min_value=1.0, value=250.0, step=10.0)
    custo_fp = col_j.number_input('Custo de bloquear um cliente bom (R$)', min_value=0.0, value=50.0, step=5.0)

    fraudes_mes = volume * (taxa / 100)
    legitimas_mes = volume - fraudes_mes
    prejuizo_sem_modelo = fraudes_mes * ticket

    resultados = []
    for nome, m in metricas['metricas'].items():
        mc = m['matriz_confusao']
        recall = mc['tp'] / (mc['tp'] + mc['fn'])
        taxa_fp = mc['fp'] / (mc['fp'] + mc['tn'])
        fraudes_barradas = fraudes_mes * recall
        bloqueios_indevidos = legitimas_mes * taxa_fp
        evitado = fraudes_barradas * ticket
        custo_bloqueios = bloqueios_indevidos * custo_fp
        resultados.append({
            'Modelo': nome,
            'Fraudes barradas': round(fraudes_barradas),
            'Prejuizo evitado': formatar_reais(evitado),
            'Clientes bons bloqueados': round(bloqueios_indevidos),
            'Custo dos bloqueios': formatar_reais(custo_bloqueios),
            'Resultado liquido': formatar_reais(evitado - custo_bloqueios),
            'liquido_numero': evitado - custo_bloqueios,
        })

    st.metric('Prejuizo mensal sem nenhum modelo', formatar_reais(prejuizo_sem_modelo))

    tabela = pd.DataFrame(resultados)
    st.dataframe(tabela.drop(columns=['liquido_numero']), hide_index=True)

    melhor_resultado = max(resultados, key=lambda r: r['liquido_numero'])
    st.success('Melhor resultado: ' + melhor_resultado['Modelo'] + ', com ganho liquido de ' + melhor_resultado['Resultado liquido'] + ' por mes.')
    st.caption('Um modelo que bloqueia muitos clientes bons pode ter resultado liquido pior mesmo pegando mais fraudes.')