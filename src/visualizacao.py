"""
src/visualizacao.py
===================

Plotagem e geração de figuras.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from uncertainties import unumpy as unp
from src.matematica import calcula_taxa_com_incerteza
from typing import List, Tuple, Optional, Dict
    
plt.rcParams.update(
    {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12}
)


# ------------------------------------------------------------------------


def plot_difracao(
    df: pd.DataFrame, 
    titulo: str, 
    offset_step: float = 1e10, 
    fig_size: Optional[Tuple[float, float]] = None, 
    usar_steps: bool = False
) -> Figure:
    
    """
    Plota as difrações empilhadas com coloração por temperatura.

    Parâmetros:
    - df_proc_final: dataframe com colunas:
        ["temperatura[K]", "step", "dados"]
    - titulo: título do gráfico
    - offset_step: espaçamento vertical
    - usar_steps: se True, eixo da direita mostra steps em vez de temperatura
    """

    altura = max(5, int(df.shape[0] * 0.5))
    if fig_size is None:
        fig_size = (13, altura)

    temperaturas_u = df["temperatura[K]"].values
    temperaturas_nom = unp.nominal_values(temperaturas_u)

    temp_min = min(temperaturas_nom)
    temp_max = max(temperaturas_nom) 
    
    norm = mcolors.Normalize(vmin=temp_min, vmax=temp_max)
    cmap = plt.colormaps.get_cmap("viridis")
    
    offset = 0
    fig = plt.figure(figsize=fig_size, dpi=600)
    ax1 = fig.add_subplot(1, 1, 1)
    
    offsets = []
    labels_y = []
    
    for i, linha in df.iterrows():
        temp = linha["temperatura[K]"]
        temp_nom = temp.nominal_value

        df = linha["dados"]
        X = df["2theta (degree)"]
        y = df["Intensity (a.u.)"]
        cor = cmap(norm(temp_nom))
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
    ax1.set_ylim(bottom=0, top=offset+(offset_step*3/2))
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

    fig.subplots_adjust(right=0.79)
    cax = fig.add_axes([0.90, 0.11, 0.04, 0.77])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Temperature (K)", labelpad=10)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())
    cbar.ax.tick_params(which="major", direction="out", length=5, width=1)
    cbar.ax.tick_params(which="minor", direction="out", length=3, width=0.8)
    
    return fig


# ------------------------------------------------------------------------


def plot_matriz_corr(
    target_temp: float, 
    tol_temp: float, 
    target_2theta: float, 
    tol_2theta: float, 
    df_proc_final: pd.DataFrame, 
    titulo: Optional[str] = None, 
    skip_steps: int = 0
) -> Figure:
    
    """
    Gera um heatmap da matriz de correlação de Pearson para faixas específicas de temperatura e ângulo.

    Parâmetros:
    - target_temp: temperatura central de interesse
    - tol_temp: tolerância para filtro de temperatura
    - target_2theta: ângulo 2-theta central de interesse
    - tol_2theta: tolerância para filtro de ângulo
    - df_proc_final: DataFrame processado com as colunas ["temperatura[K]", "step", "dados"]
    - titulo: título opcional do gráfico (gera automático se "None")
    - skip_steps: número de passos (steps) iniciais a serem desconsiderados no plot
    """

    if titulo is None:
        titulo = f"Temperatura = {target_temp} K +/- {tol_temp}°\n2θ = {target_2theta}° +/- {tol_2theta}°"

    dados_filtrados = {}

    for i, linha in df_proc_final.iterrows():
        temp = linha["temperatura[K]"]
        step =linha["step"]
        df = linha["dados"]

        if (target_temp - tol_temp) <= temp <= (target_temp + tol_temp) and step >= skip_steps:
            df_filtro = df[
                (df["2theta (degree)"] >= target_2theta - tol_2theta)
                & (df["2theta (degree)"] <= target_2theta + tol_2theta)
            ]

            dados_filtrados[f"{temp:.3f} K"] = df_filtro["Intensity (a.u.)"].values


    df_matrix = pd.DataFrame(dados_filtrados)
    corr_matrix = df_matrix.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    heatmap = sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap="viridis",
        annot=True,
        fmt=".3f",
        annot_kws={"size": 8},
        square=True,
        cbar_kws={
            "shrink": 0.92,
            "aspect": 13
        }
    )

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(0.5)
    cbar.set_label("Correlação de Pearson", labelpad=10)

    ax.set_title(
        titulo,
        pad=15,
        fontsize=12,
    )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()

    return fig


# ------------------------------------------------------------------------


def plot_temperatura_taxa(
    df: pd.DataFrame, 
    col_tempo: str = "tempo_decorrido[s]", 
    col_temp: str = "temperatura[K]", 
    titulo: str = "Temperatura e Taxa de Variação x Tempo"
) -> Figure:
    """
    Plota a variação de temperatura e sua taxa de aquecimento/resfriamento ao longo do tempo.

    Parâmetros:
    - df: DataFrame contendo os dados (ex: df_proc ou df_calib)
    - col_tempo: nome da coluna de tempo em segundos
    - col_temp: nome da coluna de temperatura em Kelvin
    - titulo: título principal do gráfico
    """
    
    X_tempos = df[col_tempo].values
    y_temps_u = df[col_temp].values
    y_temps_nom = unp.nominal_values(y_temps_u)
    y_temps_std = unp.std_devs(y_temps_u)

    taxas_u = calcula_taxa_com_incerteza(y_temps_u, X_tempos) * 60 

    taxas_nom = unp.nominal_values(taxas_u)
    taxas_std = unp.std_devs(taxas_u)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=600, sharex=True)

    # Plot 1 - Temperatura x Tempo
    ax1.plot(X_tempos, y_temps_nom, color="dodgerblue", linewidth=1.2)

    ax1.fill_between(X_tempos, y_temps_nom - y_temps_std, y_temps_nom + y_temps_std, 
                     color="dodgerblue", alpha=0.3, linewidth=0, label=r"Incerteza ($\pm 1\sigma$)")
    
    ax1.set_ylabel("Temperatura (K)", fontsize=12)
    ax1.set_title(titulo, fontsize=16)
    ax1.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.5)

    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.tick_params(axis="both", which="major", direction="out", length=8, width=1)
    ax1.tick_params(axis="both", which="minor", direction="out", length=4, width=0.7)

    # Plot 2 - Taxa de variação x Tempo
    ax2.plot(X_tempos, taxas_nom, color="crimson", linewidth=1.2)

    ax2.fill_between(X_tempos, taxas_nom - taxas_std, taxas_nom + taxas_std, 
                     color="crimson", alpha=0.3, linewidth=0, label=r"Incerteza propagada")
                     
    ax2.set_xlabel("Tempo decorrido (s)", fontsize=12)
    ax2.set_ylabel("Taxa (K/min)", fontsize=12)
    ax2.grid(which="major", linestyle="--", linewidth=0.6, alpha=0.5)

    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.tick_params(axis="both", which="major", direction="out", length=8, width=1)
    ax2.tick_params(axis="both", which="minor", direction="out", length=4, width=0.7)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------------


def plot_difracao_3d(
    df: pd.DataFrame, 
    titulo: str = "",
    fig_size: Tuple[int, int] = None,
    destacar_indicies: List[int] = None,
    cor_default: str = None,
    cor_destaque: str = None,
    fill_between: bool  = False,
    fill_between_alpha: float = 1.0,
    aspect: Tuple[float, float, float] = (1, 1, 1),
    elev: float= 25,
    azim: float = -75,
    step_linhas: int = 1,
    step_ticks_y: int = None,
    label_pad_temp: int = 55,
    label_pad_intensity: int = 55,
    ha_y: str = "left",
    ha_z: str = "left",
    labels_size=14
) -> Figure:
    
    """
    Plota os difratogramas empilhados em 3D.

    Parâmetros:
    -----------
    df : pd.DataFrame
        DataFrame contendo os dados. Deve possuir as colunas "temperatura[K]" 
        e "dados" (DataFrame interno com "2theta (degree)" e "Intensity (a.u.)").
    titulo : str
        Título a ser exibido no topo do gráfico.
    fig_size : Tuple[int, int], opcional
        Tamanho da figura gerada (largura, altura). Se None, 
        a altura será calculada com base na quantidade de linhas plotadas.
    destacar_indicies : List[int], opcional
        Lista de índices (correspondentes às linhas originais do df) 
        que devem receber uma cor de destaque no gráfico.
    cor_default : str, opcional
        Cor única e fixa para todas as linhas (ex: "black"). Se None, as linhas 
        serão coloridas de acordo com a temperatura utilizando o colormap "viridis".
    cor_destaque : str, opcional
        Cor a ser aplicada especificamente nas linhas indicadas em `destacar_indicies`.
    fill_between : bool, opcional (default=False)
        Se True, preenche a área abaixo de cada curva de difração até a intensidade mínima.
    fill_between_alpha : float, opcional (default=1.0)
        Nível de opacidade do preenchimento.
    aspect : Tuple[float, float, float], opcional (default=(1.0, 1.0, 1.0))
        Proporção geométrica da caixa do gráfico 3D nos eixos X, Y e Z.
    elev : float, opcional (default=18.0)
        Ângulo de elevação vertical (em graus) da câmera em relação ao plano 3D.
    azim : float, opcional (default=-70.0)
        Ângulo azimutal horizontal (em graus) para rotacionar o gráfico 3D.
    step_linhas : int, opcional (default=1)
        Passo dos dados de difração
    step_ticks_y : int, opcional (default=None)
        Passo dos ticks do eixo y.
    Retorna:
    --------
    Figure
        Objeto Figure do Matplotlib.
    """

    df_plot = df.iloc[::step_linhas]

    altura = max(5, int(df.shape[0] * 0.5))
    if fig_size is None:
        fig_size = (15, altura)

    temperaturas_u = df["temperatura[K]"].values
    temperaturas_nom = unp.nominal_values(temperaturas_u)

    temp_min = min(temperaturas_nom)
    temp_max = max(temperaturas_nom)
    
    norm = mcolors.Normalize(vmin=temp_min, vmax=temp_max)
    cmap = plt.colormaps.get_cmap("viridis")
    
    fig = plt.figure(figsize=fig_size, dpi=600)
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")
    ax1.set_box_aspect(aspect) 

    Z_vals = []
    y_ticks = []
    y_labels = []
    for i, linha in df_plot.iterrows():
        temp_u = linha["temperatura[K]"]
        temp_nom = temp_u.nominal_value
        temp_std = temp_u.std_dev

        dados = linha["dados"]
        X = dados["2theta (degree)"].values

        Y = np.full_like(X, temp_nom)
        Z = dados["Intensity (a.u.)"].values

        Z_vals.extend([max(Z), min(Z)])

        if cor_default is not None: 
            cor = cor_default
        else:
            cor = cmap(norm(temp_nom))

        if destacar_indicies is not None:
            if i in destacar_indicies:
                if cor_destaque is None: 
                    cor_destaque = cor
                
                ax1.plot(X, Y, Z, linewidth=1.4, color=cor_destaque, zorder=-i+1)
            else:
                ax1.plot(X, Y, Z, linewidth=1.4, color=cor, zorder=-i+1)
        else:
            ax1.plot(X, Y, Z, linewidth=1.4, color=cor, zorder=-i+1)

        if fill_between:
            ax1.fill_between(
                X, Y, Z, X, Y, min(Z), 
                facecolors="white", 
                alpha=fill_between_alpha, 
                zorder=-i
            )
        
        y_ticks.append(temp_nom)
        y_labels.append(f"{temp_nom:.1f}±{temp_std:.1f}")
    
    # Configurações do gráfico
    plt.title(titulo, y=0.986, fontsize=16)
    ax1.set_xlabel("2theta (degree)", fontsize=labels_size)
    ax1.set_ylabel("Temperature (K)", fontsize=labels_size, labelpad=label_pad_temp)
    ax1.set_zlabel("Intensity (a.u.)", fontsize=labels_size, labelpad=label_pad_intensity)

    ax1.set_xlim(min(X), max(X))
    ax1.set_ylim(temp_min, temp_max)
    ax1.set_zlim(min(Z_vals), max(Z_vals))
    
    if step_ticks_y is not None:
        ticks_plot = y_ticks[::step_ticks_y]
        labels_plot = y_labels[::step_ticks_y]
    else:
        ticks_plot = y_ticks
        labels_plot = y_labels

    ax1.set_yticks(ticks_plot)
    ax1.set_yticklabels(
        labels_plot, 
        ha=ha_y,
        va="center", 
        rotation_mode="anchor" 
    )   

    z_ticks = ax1.get_zticks()  
    ax1.set_zticks(z_ticks)

    ax1.set_zticklabels(
        [f"{tick:g}" for tick in z_ticks],
        ha=ha_z,          
        va="center", 
        rotation_mode="anchor"
    )
    
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.zaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which="major", direction="out", length=7, width=1.2)
    ax1.tick_params(which="minor", direction="out", length=4, width=0.8)

    ax1.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))

    ax1.xaxis.pane.set_edgecolor("white")
    ax1.yaxis.pane.set_edgecolor("white")
    ax1.zaxis.pane.set_edgecolor("white")

    ax1.grid(False)
    ax1.view_init(elev=elev, azim=azim)
    
    return fig


# ------------------------------------------------------------------------


def plota_dsc(
    titulo: str = "", 
    fig_size: Tuple[int, int] = (10, 6),
    df_resf: Optional[pd.DataFrame] = None, 
    df_aquec: Optional[pd.DataFrame] = None,
    range_temp_resf: Optional[Tuple[float, float]] = None,
    range_temp_aquec: Optional[Tuple[float, float]] = None,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    x_text_shifts: Optional[List[float]] = None,
    y_text_shifts: Optional[List[float]] = None,
    fontsize_legenda: int = 10,
    pontos: Optional[Dict[str, Tuple[float, float]]] = None,
    posicao_quadro: Tuple[float, float] = (0.96, 0.05)
) -> Figure:
    """
    Plota as curvas de resfriamento e aquecimento do DSC, permitindo recortes térmicos,
    anotações de pontos notáveis (Tg, Fusão, etc.) e um quadro de resumo.

    Parâmetros:
    - titulo: Título principal exibido no topo do gráfico.
    - fig_size: Dimensões da figura gerada (largura, altura).
    - df_resf: DataFrame contendo a curva de resfriamento (Temperatura na col 0, Fluxo na col 2).
    - df_aquec: DataFrame contendo a curva de aquecimento.
    - range_temp_resf: Limites (min, max) para fatiar e exibir apenas uma seção do resfriamento.
    - range_temp_aquec: Limites (min, max) para fatiar e exibir apenas uma seção do aquecimento.
    - x_lim: Limites forçados para exibição do eixo X (min, max).
    - y_lim: Limites forçados para exibição do eixo Y (min, max).
    - x_text_shifts: Lista com ajustes de distanciamento horizontal para as labels dos pontos.
    - y_text_shifts: Lista com ajustes de distanciamento vertical para as labels dos pontos.
    - fontsize_legenda: Tamanho da fonte utilizada na legenda das curvas.
    - pontos: Dicionário contendo os nomes dos eventos e suas coordenadas (Temp, Fluxo).
    - posicao_quadro: Posição (X, Y) do quadro de resumo em coordenadas relativas (0.0 a 1.0).

    Retorna:
    - fig: Objeto Figure do Matplotlib renderizado.
    """
    
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)

    if df_resf is not None:
        df_resf = df_resf = df_resf[
            (df_resf.iloc[:, 0] >= range_temp_resf[0]) & 
            (df_resf.iloc[:, 0] <= range_temp_resf[1])
        ].copy()
        X_temp_resf = df_resf.iloc[:, 0]
        y_dsc_resf = df_resf.iloc[  :, 2]

        ax.plot(X_temp_resf, y_dsc_resf, color="dodgerblue", linewidth=1.5, label="Resfriamento (10°K/min)")
    
    if df_aquec is not None:
        df_aquec = df_aquec[
            (df_aquec.iloc[:, 0] >= range_temp_aquec[0]) & 
            (df_aquec.iloc[:, 0] <= range_temp_aquec[1])
        ].copy()
        X_temp_aquec = df_aquec.iloc[:, 0]
        y_dsc_aquec = df_aquec.iloc[:, 2]

        ax.plot(X_temp_aquec, y_dsc_aquec, color="crimson", linestyle="--", linewidth=1.5, label="Aquecimento (10°K/min)")
    
    ax.set_xlabel("Temperatura (K)", fontsize=12)
    ax.set_ylabel("Fluxo de Calor (mW/mg)", fontsize=12)
    ax.set_title(titulo, fontsize=16)

    if x_lim is None:
        ax.set_xlim(
            min(min(df_resf.iloc[:, 0]), min(df_aquec.iloc[:, 0])), 
            max(max(df_aquec.iloc[:, 0]), max(df_aquec.iloc[:, 0]))
        )
    else:
        ax.set_xlim(left=x_lim[0], right=x_lim[1])

    if y_lim is None:
        ax.set_ylim(min(y_dsc_resf), max(y_dsc_aquec))
    else:
        ax.set_ylim(bottom = y_lim[0], top = y_lim[1])

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if pontos is not None:

        tamanho_max_nome = max(len(nome) for nome in pontos.keys())
        texto_resumo = ""
        i = 0

        for nome_ponto, (x_val, y_val) in pontos.items():
            
            # linha horizontal
            ax.plot([x_min, x_val], [y_val, y_val], color="gray", linestyle="--", lw=0.8)
            # linha vertical
            ax.plot([x_val, x_val], [y_min, y_val], color="gray", linestyle="--", lw=0.8)
            # ponto
            ax.scatter(x_val, y_val, color="black", s=30, zorder=10)

            if x_text_shifts is not None:
                x_shift = x_text_shifts[i]
            else: 
                x_shift = 0

            if y_text_shifts is not None:
                y_shift = y_text_shifts[i]
            else:
                y_shift = 0

            x_texto = x_val + x_shift
            y_texto = y_val + y_shift        

            ax.annotate(
                f"{nome_ponto}",
                fontsize=10,
                xy=(x_val, y_val),
                xytext=(x_texto, y_texto),
            )

            nome_com_pontos = f"{nome_ponto} ="
            texto_resumo += f"{nome_com_pontos:<{tamanho_max_nome + 1}} {x_val:>7.2f} K\n"
            i += 1

        texto_resumo = texto_resumo.strip()
        
        estilo_caixa = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=1)
        
        alinhamento_v = "bottom" if posicao_quadro[1] < 0.5 else "top"
        alinhamento_h = "right" if posicao_quadro[0] > 0.5 else "left"

        ax.text(
            posicao_quadro[0], posicao_quadro[1], 
            texto_resumo, 
            transform=ax.transAxes, 
            fontsize=10,
            verticalalignment=alinhamento_v,
            horizontalalignment=alinhamento_h,
            bbox=estilo_caixa,
            linespacing=1.5
        )

    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.legend(fontsize=fontsize_legenda)

    ax = plt.gca()  
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))  
    ax.tick_params(axis="both", which="major", length=5, width=1, direction="in")
    ax.tick_params(axis="both", which="minor", length=3, width=0.7, direction="in")

    return fig