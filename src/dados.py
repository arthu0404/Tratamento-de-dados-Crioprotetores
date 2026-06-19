"""
src/dados.py
============

Leitura, extração e organização dos dados brutos e processados 
provenientes dos diferentes equipamentos (Difração de Raios X, 
Calorimetria Exploratória Diferencial - DSC e sensores de 
temperatura). 
"""

import pandas as pd
from glob import glob
import re
import os

# ------------------------------------------------------------------------


def extrair_dados_proc(
    path_proc: str, 
    periodo_aqu: float,
) -> pd.DataFrame:
    """
    Lê arquivos CSV processados e organiza em um DataFrame.

    Parâmetros:
    - path_proc: caminho da pasta com os arquivos CSV
    - periodo_aqu: período de aquisição em segundos

    Retorna:
    - df_proc: DataFrame com colunas ["nome", "temperatura[K]", "step", "dados", "tempo_decorrido[s]"]
    """

    files_data_proc = []

    for path_arquivo_csv in glob(f"{path_proc}/*.csv"):
        try:
            df = pd.read_csv(path_arquivo_csv, skiprows=17)
            nome = os.path.basename(path_arquivo_csv).replace(".csv", "")

            partes = nome.split("_")

            temp = None
            for parte in partes:
                if "Kelvin" in parte:
                    temp = parte.replace("Kelvin", "")

            if temp is None:
                raise ValueError(f"Temperatura não encontrada: {nome}")

            match = re.search(r"_(\d+)_MERGE$", nome)
            if match:
                step = int(match.group(1))
            else:
                raise ValueError(f"Step não encontrado: {nome}")

            files_data_proc.append(
                {
                    "nome": nome,
                    "temperatura[K]": float(temp),
                    "step": step,
                    "dados": df,
                }
            )

        except Exception as e:
            print(f"Erro no arquivo {path_arquivo_csv}: {e}")

    if len(files_data_proc) == 0:
        raise ValueError(f"Nenhum arquivo válido em: {path_proc}")

    df_proc = pd.DataFrame(files_data_proc)
    df_proc = df_proc.sort_values(by="step").reset_index(drop=True)
    df_proc["tempo_decorrido[s]"] = df_proc.index * periodo_aqu

    return df_proc


# ------------------------------------------------------------------------


def extrair_tabela_calib(
    path_tabela: str,
) -> pd.DataFrame:
    """
    Lê a tabela de calibração e organiza em um DataFrame, além de fazer o cálculo do tempo decorrido.

    Parâmetros:
    - path_tabela: caminho do arquivo CSV de calibração

    Retorna:
    - df_calib: dataframe com colunas:
        ["cryojet_current_temp[K]", "Setpoint[K]", "T1[K]", "Time[h-m-s]", "tempo_decorrido[s]"]
    """
    df_calib = pd.read_csv(path_tabela)

    tempos_dt = pd.to_datetime(df_calib["Time[h-m-s]"], format="%H-%M-%S")
    tempos_decorridos = tempos_dt - tempos_dt.iloc[0]

    df_calib["tempo_decorrido[s]"] = tempos_decorridos.dt.total_seconds()

    return df_calib


# ------------------------------------------------------------------------


def extrai_dataframe_dsc(
    file_path: str
) -> pd.DataFrame:

    """
    Lê o arquivo de dados brutos exportado pelo equipamento DSC e o organiza em um DataFrame.

    A função localiza automaticamente a linha de cabeçalho onde os dados de 
    medição começam e renomeia as colunas para um formato padronizado.

    Parâmetros:
    - file_path: Caminho completo para o arquivo de texto/csv exportado pelo DSC.

    Retorna:
    - df: DataFrame contendo as curvas térmicas, tempo e segmentação com colunas padronizadas.
    """

    with open(file_path, "r", encoding="latin-1") as file:
        lines = file.readlines()

    data_start_line = None
    for i, line in enumerate(lines):
        if line.startswith("##Temp./°C;Time/min;DSC/(mW/mg);Sensit./(uV/mW);Segment"):
            data_start_line = i
            break

        
    if data_start_line is not None:
        # header = lines[:data_start_line]
        # data_lines = lines[data_start_line:]

        df = pd.read_csv(file_path, skiprows=data_start_line, encoding="latin-1", sep=';')

        df = df.rename(columns={
            "##Temp./°C": "Temperatura (°C)",
            "Time/min": "Tempo (min)",
            "DSC/(mW/mg)": "DSC (mW/mg)",
            "Gas Flow(purge2)/(ml/min)": "Gas Flow - purge 2 (ml/min)",
            "Gas Flow(protective)/(ml/min)": "Gas Flow - protective (ml/min)",
            "Sensit./(uV/mW)": "Sensibilidade (uV/mW)",
            "Segment": "Segmento"
        })

        return df
