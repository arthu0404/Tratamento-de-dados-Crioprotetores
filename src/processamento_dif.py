"""
src/processamento.py
====================

Processamento e tratamento dos dados de Difração de Raios X (DRX) e 
perfis térmicos. 
"""

import numpy as np
import pandas as pd
from uncertainties import ufloat
from uncertainties import unumpy as unp
from typing import List, Tuple, Dict, Union
from src.matematica import calcula_u_std


# ------------------------------------------------------------------------


def aplicar_inc_df(
    df: pd.DataFrame, 
    cols: List[str], 
    incertezas: List[List[float]]   
) -> pd.DataFrame:
    
    """
    Aplica incertezas aos valores de colunas específicas de um DataFrame, 
    transformando os dados em objetos do tipo ufloat.

    Parâmetros:
    - df: DataFrame original contendo os dados
    - cols: lista com os nomes das colunas (strings) que receberão as incertezas
    - incertezas: lista de listas contendo os valores de incerteza correspondentes a cada coluna

    Retorna:
    - u_df: cópia do DataFrame original com os valores das colunas especificadas 
      substituídos por objetos ufloat (valor +/- incerteza)
    """
    u_df = df.copy()
    
    for col, inc_list in zip(cols, incertezas):
        u_df[col] = [ufloat(val, inc) for val, inc in zip(df[col], inc_list)]
        
    return u_df


# ------------------------------------------------------------------------


def corrigir_anomalia(
    targets_2theta: Union[float, List[float]], 
    tol_2theta: float, 
    idx_range: int, 
    df_proc: pd.DataFrame
) -> pd.DataFrame:  
    
    """
    Substitui uma região de anomalia por uma interpolação linear.

    Parâmetros:
    - target_2theta: valor central de 2-theta onde a anomalia se encontra
    - tol_2theta: tolerância para busca do pico máximo da anomalia
    - idx_range: número de índices para antes e depois do pico para aplicar a correção
    - df_proc: DataFrame processado contendo os DataFrames internos em 'dados'

    Retorna:
    - df_proc_tratado: DataFrame com as intensidades corrigidas na região especificada
    """
    
    if not isinstance(targets_2theta, (list, tuple)):
        targets_2theta = [targets_2theta]

    df_proc_tratado = df_proc.copy()
    
    for i, linha in df_proc.iterrows():
        df_interno = linha["dados"].copy()

        for target in targets_2theta:
            mask = (df_interno["2theta (degree)"] >= target - tol_2theta) & \
                   (df_interno["2theta (degree)"] <= target + tol_2theta)
        
            idx_max = df_interno.loc[mask, "Intensity (a.u.)"].idxmax()
            pos_max = df_interno.index.get_loc(idx_max)

            if idx_range - 1 < pos_max < len(df_interno) - idx_range:
                idx_inicio, idx_fim = pos_max - idx_range, pos_max + idx_range

                val_antes = df_interno.iloc[idx_inicio]["Intensity (a.u.)"]
                val_depois = df_interno.iloc[idx_fim]["Intensity (a.u.)"]

                n_pontos = idx_fim - idx_inicio + 1
                novos_vals = np.linspace(val_antes, val_depois, n_pontos)

                col_idx = df_interno.columns.get_loc("Intensity (a.u.)")
                    
                df_interno.iloc[idx_inicio : idx_fim + 1, col_idx] = novos_vals

        df_proc_tratado.at[i, "dados"] = df_interno


    return df_proc_tratado


# ------------------------------------------------------------------------


def calc_erro_termopar(
    T: float
) -> float:
    
    """
    Calcula a tolerância/erro associada à leitura de um termopar padrão 
    em função da temperatura medida.
    
    Parâmetros:
    - T: Temperatura medida no experimento.

    Retorna:
    - erro: Valor absoluto da incerteza máxima esperada (float) segundo as 
      regras de tolerância do termopar.
    """

    if T > 0:
        return max(2.2, abs(0.0075 * T))
    else:
        return max(2.2, abs(0.02 * T))


# ------------------------------------------------------------------------


def separar_curvas(
    df: pd.DataFrame, 
    coluna_temp: str
) -> Tuple[pd.DataFrame, pd.DataFrame, int, float]:
    
    """
    Separa os dados das curvas em resfriamento e aquecimento.

    Parâmetros:
    - df: DataFrame com dados que deseja separar
    - coluna_temp: nome da coluna de temperatura

    Retorna:
    - df_resf: curva do resfriamento
    - df_aquec: curva do aquecimento
    - idx_min: índice da temperatura mínima
    - temp_min: valor da temperatura mínima
    """

    idx_min = unp.nominal_values(df[coluna_temp]).argmin()

    df_resf = df[:idx_min].copy()
    df_aquec = df[idx_min:].copy()

    temp_min = df[coluna_temp].iloc[idx_min]

    return (df_resf, df_aquec, idx_min, temp_min)


# ------------------------------------------------------------------------


def alinhar_por_temperatura(
    df_proc: pd.DataFrame, 
    df_calib: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Alinha dados experimentais com calibração pela temperatura.

    Parâmetros:
    - df_proc: DataFrame processado
    - df_calib: DataFrame de calibração

    Retorna:
    - df_temp_corr: DataFrame contendo:
        ["t_decorrido_proc", "t_decorrido_calib", "temp_t1", "temp_cryojet_calib", "temp_cryojet_proc", "diff_temp_cryojet", "diff_t"]
    - estatisticas: dicionário com:
        ["media_diff_temp", "std_diff_temp", "media_diff_t", "std_diff_t"]
    """

    idx_min_temp_calib = unp.nominal_values(df_calib["cryojet_current_temp[K]"]).argmin()
    df_calib_resf = df_calib.iloc[:idx_min_temp_calib].copy()
    df_calib_aquec = df_calib.iloc[idx_min_temp_calib:].copy()

    idx_min_temp_proc = unp.nominal_values(df_proc["temperatura[K]"]).argmin()

    dados = []
    t0_calib = None

    for i in range(len(df_proc)):
        t_proc = df_proc["tempo_decorrido[s]"].iloc[i]
        temp_proc = df_proc["temperatura[K]"].iloc[i]

        if i <= idx_min_temp_proc:
            df_ref = df_calib_resf
        else:
            df_ref = df_calib_aquec

        idx_prox = np.abs(
            unp.nominal_values(df_ref["cryojet_current_temp[K]"] - temp_proc)
        ).argmin()

        linha_calib = df_ref.iloc[idx_prox]

        if t0_calib is None:
            t0_calib = linha_calib["tempo_decorrido[s]"]

        t_calib = linha_calib["tempo_decorrido[s]"] - t0_calib
        temp_calib = linha_calib["cryojet_current_temp[K]"]
        temp_t1 = linha_calib["T1[K]"]

        val_temp_calib_decres_u  = temp_calib.nominal_value - temp_calib.std_dev
        val_temp_calib_acres_u  = temp_calib.nominal_value + temp_calib.std_dev
        
        val_temp_t1_decres_u = temp_t1.nominal_value - temp_calib.std_dev
        val_temp_t1_acres_u = temp_t1.nominal_value + temp_calib.std_dev

        houve_sobreposicao = (val_temp_calib_decres_u <= val_temp_t1_acres_u) and \
                             (val_temp_calib_acres_u >= val_temp_t1_decres_u)
        
        u_sobreposto = max(val_temp_calib_acres_u, val_temp_t1_acres_u) - \
                       min(val_temp_calib_decres_u, val_temp_t1_decres_u)
        
        dados.append({
            "t_decorrido_proc": t_proc,
            "t_decorrido_calib": t_calib,
            "temp_t1": temp_t1,
            "temp_cryojet_calib": temp_calib,
            "temp_cryojet_proc": temp_proc,
            "u_sobreposto": u_sobreposto,
            "sobreposicao_u": houve_sobreposicao,
            "diff_temp_cryojet_calib_t1": temp_calib - temp_t1,
            "diff_temp_cryojet": temp_proc - temp_calib,
            "diff_t": t_proc - t_calib,
        })

    df_temp_corr = pd.DataFrame(dados)

    media_diff_temp = np.mean(df_temp_corr["diff_temp_cryojet"].values)
    std_diff_temp = calcula_u_std(df_temp_corr["diff_temp_cryojet"].values)

    media_diff_t = np.mean(df_temp_corr["diff_t"].values)
    std_diff_t = calcula_u_std(df_temp_corr["diff_t"].values)

    estatisticas = {
        "media_diff_temp": media_diff_temp,
        "std_diff_temp": std_diff_temp,
        "media_diff_t": media_diff_t,
        "std_diff_t": std_diff_t,
    }

    return (df_temp_corr, estatisticas)


# ------------------------------------------------------------------------


def substitui_inc_df(
    df: pd.DataFrame, 
    cols: List[str],
    incertezas: List[List[float]]
) -> pd.DataFrame:
    
    """
    Substitui os valores de incerteza em colunas que já contêm objetos ufloat,
    preservando os valores nominais (temperaturas/tempos).

    Parâmetros:
    - df: DataFrame original contendo colunas com ufloats que precisam ser atualizadas.
    - cols: Lista com os nomes das colunas que terão suas incertezas sobrescritas.
    - incertezas: Lista de listas contendo as novas incertezas.

    Retorna:
    - u_df: DataFrame com as colunas atualizadas com os novos objetos ufloat.
    """

    u_df = df.copy()
    
    for col, inc_list in zip(cols, incertezas):
        u_df[col] = [ufloat(u_val.nominal_value, inc) for u_val, inc in zip(df[col], inc_list)]
        
    return u_df