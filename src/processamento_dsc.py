"""
src/processamento_dsc.py
========================

Processamento de dados obtidos por Calorimetria Exploratória 
Diferencial (DSC). 
"""

import numpy as np
import pandas as pd
from typing import Tuple 


def convert_Celc2Kelvin(
    df: pd.DataFrame,
    col: str
) -> pd.DataFrame:
    """
    Converte a temperatura de Celsius para Kelvin em uma coluna específica do DataFrame.

    Parâmetros:
    - df: DataFrame contendo a coluna de temperatura em graus Celsius (°C).
    - col: Nome da coluna que será convertida.

    Retorna:
    - df_corr: Cópia do DataFrame com os valores convertidos e o nome da 
      coluna modificado automaticamente para refletir a unidade em Kelvin (K).
    """

    df_corr = df.copy()
    df_corr[col] = df[col].values + 273.15
    nova_col = col.replace("°C", "K")
    df_corr = df_corr.rename(columns={col: nova_col})

    return df_corr


# ------------------------------------------------------------------------


def separa_aquec_resf(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa o DataFrame padronizado do DSC em duas curvas distintas: 
    resfriamento e aquecimento.

    A função se baseia na última coluna ("Segmento"), considerando 1 para 
    resfriamento e 2 para aquecimento.

    Parâmetros:
    - df: DataFrame bruto/tratado do DSC contendo a coluna de Segmento.

    Retorna:
    - df_resf: DataFrame contendo apenas os dados da etapa de resfriamento.
    - df_aquec: DataFrame contendo apenas os dados da etapa de aquecimento.
    """

    df_resf = df[df.iloc[:, -1] == 1]
    df_aquec = df[df.iloc[:, -1] == 2]

    return df_resf, df_aquec


# ------------------------------------------------------------------------


def calc_pico_dsc(
    df,
    pontos: tuple,
    tipo: str,
    col_temp: int = 0,
    col_dsc: int = 2,
) -> dict:
    
    """
    Calcula T_onset, T_pico, T_endset e entalpia de um pico de DSC.
    Convenção: exo down  →  tipo="exo" (pico negativo)
                            tipo="endo" (pico positivo)
    Parâmetros
    ----------
    df : pd.DataFrame
    pontos : ([T_min_pre, T_max_pre], [T_min_pos, T_max_pos])
        Baselines antes e depois do pico. O pico é a região entre elas.
    tipo : "exo" ou "endo"
    col_temp, col_dsc : índices das colunas
    """

    T  = df.iloc[:, col_temp].values
    HF = df.iloc[:, col_dsc].values

    range_pre, range_pos = pontos

    def _ajusta_reta(rng):
        m = (T >= rng[0]) & (T <= rng[1])
        if m.sum() < 2:
            raise ValueError(f"Poucos pontos na faixa {rng}.")
        return np.polyfit(T[m], HF[m], 1)

    def _intersecao(a1, b1, a2, b2):
        if np.isclose(a1, a2):
            raise ValueError("Retas paralelas — ajuste as faixas.")
        x = (b2 - b1) / (a1 - a2)
        return x, a1 * x + b1

    a_pre, b_pre = _ajusta_reta(range_pre)
    a_pos, b_pos = _ajusta_reta(range_pos)

    T_pico_min = range_pre[1]
    T_pico_max = range_pos[0]
    m_pico = (T >= T_pico_min) & (T <= T_pico_max)
    T_p, HF_p = T[m_pico], HF[m_pico]

    if len(T_p) < 3:
        raise ValueError("Poucos pontos na região do pico. Ajuste os intervalos.")

    idx_ext = np.argmin(HF_p) if tipo == "exo" else np.argmax(HF_p)
    T_pico_val = T_p[idx_ext]
    HF_pico_val = HF_p[idx_ext]
    
    n = len(T_p)
    quarto = max(2, n // 4)

    # tangente no lado esquerdo do pico
    a_sub, b_sub = np.polyfit(T_p[:quarto], HF_p[:quarto], 1)

    # tangente no lado direito do pico
    a_desc, b_desc = np.polyfit(T_p[-quarto:], HF_p[-quarto:], 1)

    T_onset,  _ = _intersecao(a_pre, b_pre, a_sub,  b_sub)
    T_endset, _ = _intersecao(a_pos, b_pos, a_desc, b_desc)

    # baseline = np.interp(T_p, [T_onset, T_endset],
    #                      [a_pre * T_onset + b_pre,
    #                       a_pos * T_endset + b_pos])
    # sinal = -1 if tipo == "exo" else 1
    # entalpia = sinal * np.trapezoid(HF_p - baseline, t_p) * 10  # mW/mg × K = mJ/mg

    return {
        "T_pico":   T_pico_val,
        "HF_pico":  HF_pico_val,
        "T_onset":  T_onset,
        "T_endset": T_endset,
    }


# ------------------------------------------------------------------------


def calc_tg_half_height(
    df,
    pontos: tuple,
    col_temp: int = 0,
    col_dsc: int = 2,
) -> dict:
    """
    Calcula a Tg pelo método half-height (TA Instruments, TA443).

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com os dados DSC (segmento único: aquecimento ou resfriamento).
    pontos : tuple de dois intervalos
        ([T_min_glassy, T_max_glassy], [T_min_rubbery, T_max_rubbery])
        A região de inflexão é inferida automaticamente entre os dois intervalos.
    col_temp : int
        Índice da coluna de temperatura (padrão 0).
    col_dsc : int
        Índice da coluna de fluxo de calor (padrão 2).
    """

    T  = df.iloc[:, col_temp].values
    HF = df.iloc[:, col_dsc].values

    range_glassy, range_rubbery, range_inflexao = pontos

    # range_inflexao = (
    #     min(range_glassy[1], range_rubbery[1]),
    #     max(range_glassy[0], range_rubbery[0]),
    # )

    def _ajusta_reta(rng):
        m = (T >= rng[0]) & (T <= rng[1])
        if m.sum() < 2:
            raise ValueError(f"Poucos pontos na faixa {rng}. Ajuste o intervalo.")
        return np.polyfit(T[m], HF[m], 1)  # retorna (a, b): HF = a*T + b

    def _intersecao(a1, b1, a2, b2):
        if np.isclose(a1, a2):
            raise ValueError("Retas paralelas — ajuste as faixas de temperatura.")
        
        x = (b2 - b1) / (a1 - a2)
        return x, a1 * x + b1

    def _interpola_Tg(hf_alvo):
        m = (T >= min(T_ons, T_end)) & (T <= max(T_ons, T_end))
        T_seg, HF_seg = T[m], HF[m]
        cruzamentos = np.where(np.diff(np.sign(HF_seg - hf_alvo)))[0]

        if len(cruzamentos) == 0:
            raise ValueError(
                f"Curva DSC não cruza HF_mid={hf_alvo:.4f} entre T_ons e T_end."
                "Verifique os intervalos."
            )
        
        idx = cruzamentos[len(cruzamentos) // 2]
        T0, T1   = T_seg[idx], T_seg[idx + 1]
        HF0, HF1 = HF_seg[idx], HF_seg[idx + 1]
        return T0 + (hf_alvo - HF0) * (T1 - T0) / (HF1 - HF0)

    a_gl, b_gl = _ajusta_reta(range_glassy)
    a_ru, b_ru = _ajusta_reta(range_rubbery)
    a_in, b_in = _ajusta_reta(range_inflexao)

    T_ons, _ = _intersecao(a_gl, b_gl, a_in, b_in)
    T_end, _ = _intersecao(a_ru, b_ru, a_in, b_in)

    HF_ons = a_gl * T_ons + b_gl
    HF_end = a_ru * T_end + b_ru
    HF_mid = (HF_ons + HF_end) / 2.0

    Tg = _interpola_Tg(HF_mid)

    return {
        "Tg": Tg,
        "T_ons": T_ons,
        "T_end": T_end,
        "HF_ons": HF_ons,
        "HF_end": HF_end,
        "HF_mid": HF_mid,
    }


# ------------------------------------------------------------------------


def gerar_tabela_resultados(
    resultados_gerais: dict
) -> pd.DataFrame:
    """
    Converte um dicionário mestre contendo os resultados de múltiplas amostras 
    (CPAs) em um único DataFrame do Pandas formatado.

    Parâmetros:
    - resultados_gerais: Dicionário onde as chaves são os nomes das amostras e 
      os valores são os dicionários de resultados calculados para aquela amostra.
      Ex: {"Vs55": resultados_vs55, "Glicerol": resultados_gly}

    Retorna:
    - df_tabela: DataFrame formatado com a coluna "Amostra" adicionada.
    """
    
    linhas_tabela = []
    
    for amostra, eventos_da_amostra in resultados_gerais.items():
        
        for evento, dados in eventos_da_amostra.items():
            linha = {
                "Amostra": amostra,
                "Evento": evento,
                "Temp. Característica (K)": None,
                "Onset (K)": None,
                "Endset (K)": None,
                "Fluxo de Calor (mW/mg)": None
            }
            
            if "Tg" in dados:
                linha["Temp. Característica (K)"] = dados["Tg"]
                linha["Onset (K)"] = dados["T_ons"]
                linha["Endset (K)"] = dados["T_end"]
                linha["Fluxo de Calor (mW/mg)"] = dados["HF_mid"]
                
            elif "T_pico" in dados:
                linha["Temp. Característica (K)"] = dados["T_pico"]
                linha["Onset (K)"] = dados["T_onset"]
                linha["Endset (K)"] = dados["T_endset"]
                linha["Fluxo de Calor (mW/mg)"] = dados["HF_pico"]
                
            linhas_tabela.append(linha)
            
    df_tabela = pd.DataFrame(linhas_tabela)
    
    return df_tabela.round(2)