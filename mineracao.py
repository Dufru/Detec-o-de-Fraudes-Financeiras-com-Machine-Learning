import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

from src.features import carregar_dados, preparar_dados, balancear_treino


def avaliar(modelo, x_test, y_test):
    prob = modelo.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    resultado = {
        'precisao': precision_score(y_test, pred, zero_division=0),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'pr_auc': average_precision_score(y_test, prob),
        'roc_auc': roc_auc_score(y_test, prob),
        'acuracia': accuracy_score(y_test, pred),
        'matriz_confusao': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
    }
    return resultado, prob


def main(caminho):
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    df = carregar_dados(caminho)
    x_train, x_test, y_train, y_test, scaler = preparar_dados(df)
    print('Treino:', len(x_train), 'linhas | Teste:', len(x_test), 'linhas')
    print('Fraudes no teste:', int(y_test.sum()))

    x_res, y_res = balancear_treino(x_train, y_train)
    print('Treino depois do SMOTE:', len(x_res), 'linhas')

    modelos = {
        'Regressao Logistica': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
    }

    metricas = {}
    probabilidades = {}
    treinados = {}

    for nome, modelo in modelos.items():
        print('\nTreinando', nome)
        modelo.fit(x_res, y_res)
        resultado, prob = avaliar(modelo, x_test, y_test)
        metricas[nome] = resultado
        probabilidades[nome] = prob
        treinados[nome] = modelo

        print('Precisao:', round(resultado['precisao'], 3))
        print('Recall:', round(resultado['recall'], 3))
        print('F1:', round(resultado['f1'], 3))
        print('PR-AUC:', round(resultado['pr_auc'], 3))
        print('Acuracia:', round(resultado['acuracia'], 4))
        print('Matriz de confusao:', resultado['matriz_confusao'])

    melhor = None
    melhor_pr_auc = -1
    for nome in metricas:
        if metricas[nome]['pr_auc'] > melhor_pr_auc:
            melhor_pr_auc = metricas[nome]['pr_auc']
            melhor = nome
    print('\nMelhor modelo:', melhor)

    pacote = {
        'modelo': treinados[melhor],
        'scaler': scaler,
        'colunas': list(x_train.columns),
        'nome': melhor,
    }
    joblib.dump(pacote, 'models/melhor_modelo.joblib')

    with open('reports/metricas.json', 'w', encoding='utf-8') as arquivo:
        json.dump({'melhor_modelo': melhor, 'metricas': metricas}, arquivo, indent=2)

    plt.figure(figsize=(7, 5))
    for nome in probabilidades:
        precisao, recall, _ = precision_recall_curve(y_test, probabilidades[nome])
        plt.plot(recall, precisao, label=nome)
    plt.xlabel('Recall')
    plt.ylabel('Precisao')
    plt.title('Curva Precisao-Recall')
    plt.legend()
    plt.savefig('reports/curva_precisao_recall.png')
    plt.close()

    plt.figure(figsize=(7, 5))
    for nome in probabilidades:
        fpr, tpr, _ = roc_curve(y_test, probabilidades[nome])
        auc = metricas[nome]['roc_auc']
        plt.plot(fpr, tpr, label=nome + ' (AUC = ' + str(round(auc, 3)) + ')')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Aleatorio')
    plt.xlabel('Taxa de falsos positivos')
    plt.ylabel('Taxa de verdadeiros positivos (Recall)')
    plt.title('Curva ROC')
    plt.legend()
    plt.savefig('reports/curva_roc.png')
    plt.close()

    print('Arquivos salvos nas pastas models e reports')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dados', default='data/creditcard.csv')
    args = parser.parse_args()
    main(args.dados)