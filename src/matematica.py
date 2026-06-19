"""
src/matematica_inc.py
=====================

Contém funções matemáticas e estatísticas de suporte. Centrado 
especialmente na propagação de incertezas, cálculo de derivadas 
(taxas de variação) utilizando métodos de diferenças finitas e 
ajustes de regressão linear.
"""

import numpy as np
from uncertainties.umath import sqrt
from typing import List, Union, Tuple

    
# ------------------------------------------------------------------------


def calcula_u_std(
    u_lista: Union[List[float], np.ndarray], 
    ddof: int = 1 
) -> float:
    
    """
    Calcula o desvio padrão de uma lista ou array de valores.
    
    Parâmetros:
    - u_lista: Lista ou array numpy contendo os valores.
    - ddof: Graus de liberdade. O padrão é 1 (desvio padrão amostral).
    
    Retorna:
    - raiz: O desvio padrão calculado.
    """

    media = np.mean(u_lista)
    soma_quadratica = np.sum((u_lista - media)**2)

    if ddof == 1:
        var = soma_quadratica / (len(u_lista) - 1)
    else:
        var = soma_quadratica / len(u_lista)

    raiz = sqrt(var)
    return raiz


# ------------------------------------------------------------------------


def calcula_taxa_com_incerteza(
    y_u: Union[List[float], np.ndarray], 
    x: Union[List[float], np.ndarray]
) -> np.ndarray:
    
    """
    Calcula a taxa de variação (derivada primeira) de y em relação a x, 
    propagando automaticamente as incertezas de y utilizando diferenças finitas.

    A função aplica diferenças finitas centrais para o miolo dos dados e 
    diferenças finitas laterais (forward/backward) nas bordas.

    Parâmetros:
    - y_u: Lista ou array dos valores dependentes (temperaturas com ufloat).
    - x: Lista ou array dos valores independentes (tempos).

    Retorna:
    - grad: Array numpy contendo as taxas de variação com as incertezas propagadas.
    """

    y_u = np.array(list(y_u))
    x = np.asarray(x, dtype=float)
    n = len(y_u)
    
    grad = np.empty(n, dtype=object)
    
    if n < 3:
        grad[0] = (y_u[1] - y_u[0]) / (x[1] - x[0])
        grad[-1] = grad[0]
        return grad
    
    h_d = x[1:-1] - x[:-2]
    h_s = x[2:] - x[1:-1]
    
    w_prev = -h_s / (h_d * (h_d + h_s))
    w_curr = (h_s - h_d) / (h_d * h_s)
    w_next = h_d / (h_s * (h_d + h_s))
    
    grad[1:-1] = w_prev * y_u[:-2] + w_curr * y_u[1:-1] + w_next * y_u[2:]
    
    # Borda esquerda (i = 0)
    h0 = x[1] - x[0]
    h1 = x[2] - x[1]
    w0_0 = -(2 * h0 + h1) / (h0 * (h0 + h1))
    w1_0 = (h0 + h1) / (h0 * h1)
    w2_0 = -h0 / (h1 * (h0 + h1))
    grad[0] = w0_0 * y_u[0] + w1_0 * y_u[1] + w2_0 * y_u[2]
    
    # Borda direita (i = n-1)
    hn2 = x[-1] - x[-2]
    hn3 = x[-2] - x[-3]
    w0_n = hn2 / (hn3 * (hn2 + hn3))
    w1_n = -(hn2 + hn3) / (hn2 * hn3)
    w2_n = (2 * hn2 + hn3) / (hn2 * (hn2 + hn3))
    grad[-1] = w0_n * y_u[-3] + w1_n * y_u[-2] + w2_n * y_u[-1]
    
    return grad


# ------------------------------------------------------------------------


def calc_linreg(
    x: Union[List[float], np.ndarray], 
    y: Union[List[float], np.ndarray]
) -> Tuple[float, float]:
    """
    Calcula a regressão linear de um conjunto de dados utilizando o método dos mínimos quadrados.

    Parâmetros:
    - x: Lista ou array com os valores independentes (ex: temperaturas).
    - y: Lista ou array com os valores dependentes (ex: fluxo de calor).

    Retorna:
    - a: Coeficiente angular da reta (inclinação).
    - b: Coeficiente linear da reta (interseção com o eixo y).
    """
    
    a, b = np.polyfit(x, y, 1)
    return a, b
