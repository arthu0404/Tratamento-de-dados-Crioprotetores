# Tratamento e análise de dados de crioprotetores (CPAs)

```bash
tratamento-analise-dados-crioprotetores
├── dados/
│   ├── 01_brutos/
│   │   ├── calibracao/
│   │   ├── difracao/
│   │   │   └── main_folder_vs55/
│   │   │       └── vs55/
│   │   └── dsc/
│   └── 02_processados/
│       ├── calibracao/
│       ├── high_throughput/
│       │   └── merge_files/
│       └── reproc_july_2025/
│           ├── main_folder_capilar_vazio_girando_2_3/
│           ├── main_folder_capilar_vazio_girando_6kpmin/
│           :
│           :
│               └── (Cada pasta acima contém seu respectivo /merge_files/)
│
├── notebooks/
│   ├── 01_extracao_dados_calib_temp.ipynb # Gera .csv de calibração
│   ├── 02_analise_e_plot_difracao.ipynb   # Análise e tratamento dos dados de DRX
│   └── 03_analise_e_plot_dsc.ipynb        # Análise dos dados de DSC
│
├── resultados/
│   ├── dsc/
│   └── rampas/
│
├── src/
│   ├── dados.py                 # Extração de dados
│   ├── matematica.py            # Funções matemáticas
│   ├── parametros.py            # Constantes do projeto
│   ├── processamento_dif.py     # Processamento de DRX
│   ├── processamento_dsc.py     # Processamento de DSC
│   └── visualizacao.py          # Funções de plotagem
│
└── main.py                      # Script principal com o pipeline para difração
```