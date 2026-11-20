import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def carregar_dados(caminho):
    df = pd.read_csv(caminho)
    return df


def preparar_dados(df):
    df = df.copy()
    df['Hour'] = (df['Time'] / 3600) % 24
    df = df.drop(columns=['Time'])

    x = df.drop(columns=['Class'])
    y = df['Class']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    x_train = x_train.copy()
    x_test = x_test.copy()

    scaler = StandardScaler()
    x_train[['Amount', 'Hour']] = scaler.fit_transform(x_train[['Amount', 'Hour']])
    x_test[['Amount', 'Hour']] = scaler.transform(x_test[['Amount', 'Hour']])

    return x_train, x_test, y_train, y_test, scaler


def balancear_treino(x_train, y_train):
    qtd_fraudes = int(y_train.sum())
    vizinhos = min(5, qtd_fraudes - 1)
    if vizinhos < 1:
        vizinhos = 1

    smote = SMOTE(random_state=42, k_neighbors=vizinhos)
    x_res, y_res = smote.fit_resample(x_train, y_train)
    return x_res, y_res