import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
import os

def extrair_dados_proc(path_proc, periodo_aqu):
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
        df = pd.read_csv(path_arquivo_csv, skiprows=17  )
        nome = os.path.basename(path_arquivo_csv).replace(".csv", "")

        partes = nome.split("_")
        for parte in partes:
            if "Kelvin" in parte:
                temp = parte.replace("Kelvin", "")

        step = int(nome.split("_")[-2])

        files_data_proc.append(
            {"nome": nome, "temperatura[K]": float(temp), "step": step, "dados": df}
        )

    df_proc = pd.DataFrame(files_data_proc)
    df_proc = df_proc.sort_values(by="step").reset_index(drop=True)
    df_proc["tempo_decorrido[s]"] = df_proc.index * periodo_aqu

    return df_proc

# ------------------------------------------------------------------------

def corrigir_anomalia(target_2theta, tol_2theta, idx_range, df_proc):
    df_proc_tratado = df_proc.copy()
    
    for i, linha in df_proc.iterrows():
        df_interno = linha["dados"]

        mask = (df_interno["2theta (degree)"] >= target_2theta - tol_2theta) & \
            (df_interno["2theta (degree)"] <= target_2theta + tol_2theta)
    
        idx_max = df_interno.loc[mask, "Intensity (a.u.)"].idxmax()
        pos_max = df_interno.index.get_loc(idx_max)

        if idx_range - 1 < pos_max < len(df_interno) - idx_range:
            idx_inicio, idx_fim =  pos_max - idx_range, pos_max + idx_range

            val_antes = df_interno.iloc[idx_inicio]["Intensity (a.u.)"]
            val_depois = df_interno.iloc[idx_fim]["Intensity (a.u.)"]

            n_pontos = idx_fim - idx_inicio + 1
            novos_vals = np.linspace(val_antes, val_depois, n_pontos)

            col_idx = df_interno.columns.get_loc("Intensity (a.u.)")
                
            df_interno.iloc[idx_inicio : idx_fim + 1, col_idx] = novos_vals

            df_proc_tratado.at[i, "dados"] = df_interno

    return df_proc_tratado

# ------------------------------------------------------------------------

def extrair_tabela_calib(path_tabela):
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


def separar_curvas(df, coluna_temp):
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

    idx_min = df[coluna_temp].argmin()

    df_resf = df[:idx_min].copy()
    df_aquec = df[idx_min:].copy()

    temp_min = df[coluna_temp].iloc[idx_min]

    return df_resf, df_aquec, idx_min, temp_min

# ------------------------------------------------------------------------

def alinhar_por_temperatura(df_proc, df_calib):
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

    idx_min_temp_calib = df_calib["cryojet_current_temp[K]"].argmin()
    df_calib_resf = df_calib.iloc[:idx_min_temp_calib].copy()
    df_calib_aquec = df_calib.iloc[idx_min_temp_calib:].copy()

    idx_min_temp_proc = df_proc["temperatura[K]"].argmin()

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
            df_ref["cryojet_current_temp[K]"] - temp_proc
        ).idxmin()

        linha_calib = df_ref.loc[idx_prox]

        if t0_calib is None:
            t0_calib = linha_calib["tempo_decorrido[s]"]

        t_calib = linha_calib["tempo_decorrido[s]"] - t0_calib
        temp_calib = linha_calib["cryojet_current_temp[K]"]
        temp_t1 = linha_calib["T1[K]"]

        dados.append({
            "t_decorrido_proc": t_proc,
            "t_decorrido_calib": t_calib,
            "temp_t1": temp_t1,
            "temp_cryojet_calib": temp_calib,
            "temp_cryojet_proc": temp_proc,
            "diff_temp_cryojet": temp_proc - temp_calib,
            "diff_t": t_proc - t_calib
        })

    df_temp_corr = pd.DataFrame(dados)

    media_diff_temp = np.mean(df_temp_corr["diff_temp_cryojet"])
    std_diff_temp = np.std(df_temp_corr["diff_temp_cryojet"])

    media_diff_t = np.mean(df_temp_corr["diff_t"])
    std_diff_t = np.std(df_temp_corr["diff_t"])

    estatisticas = {
        "media_diff_temp": media_diff_temp,
        "std_diff_temp": std_diff_temp,
        "media_diff_t": media_diff_t,
        "std_diff_t": std_diff_t,
    }

    return df_temp_corr, estatisticas

# ------------------------------------------------------------------------

def plot_difracao(df_proc_final, titulo, offset_step=1e10, usar_steps=False):
    """
    Plota as difrações empilhadas com coloração por temperatura.

    Parâmetros:
    - df_proc_final: dataframe com colunas:
        ["temperatura[K]", "step", "dados"]
    - titulo: título do gráfico
    - offset_step: espaçamento vertical
    - usar_steps: se True, eixo da direita mostra steps em vez de temperatura
    """

    temperaturas = df_proc_final["temperatura[K]"].values
    temp_min = min(temperaturas)
    temp_max = max(temperaturas)
    
    norm = mcolors.Normalize(vmin=temp_min, vmax=temp_max)
    cmap = plt.colormaps.get_cmap("viridis")
    
    offset = 0
    fig = plt.figure(figsize=(13, 20), dpi=600)
    ax1 = fig.add_subplot(1, 1, 1)
    
    offsets = []
    labels_y = []
    
    for i, linha in df_proc_final.iterrows():
        temp = linha["temperatura[K]"]
        df = linha["dados"]
        X = df["2theta (degree)"]
        y = df["Intensity (a.u.)"]
        cor = cmap(norm(temp))
        ax1.plot(X, y + offset, linewidth=0.9, color=cor)
        
        offsets.append(offset)
        if usar_steps:
            labels_y.append(str(linha["step"]))
        else:
            labels_y.append(f"{temp:.3f}")
        offset += offset_step
    
    # Configurações do gráfico
    plt.title(titulo, fontsize=16)
    ax1.set_xlabel("2theta (degree)")
    ax1.set_ylabel("Intensity + offset (a.u.)")
    ax1.set_xlim(0)
    ax1.set_ylim(bottom=0, top=offset)
    ax1.set_yticks([])
    
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="major", direction="out", length=7, width=1.2)
    ax1.tick_params(which="minor", direction="out", length=4, width=0.8)
    ax1.grid(which="major", linestyle="--", linewidth=0.5, alpha=0.4)
    
    # Eixo y à direita
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(offsets)
    ax2.set_yticklabels(labels_y, fontsize=10)
    if usar_steps:
        ax2.set_ylabel("Steps (a.u.)", labelpad=10)
    else:
        ax2.set_ylabel("Temperature [K]", labelpad=10)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax1, pad=0.12, aspect=25)
    cbar.set_label("Temperature (K)", labelpad=10)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())
    cbar.ax.tick_params(which="major", direction="out", length=5, width=1)
    cbar.ax.tick_params(which="minor", direction="out", length=3, width=0.8)
    
    plt.show()