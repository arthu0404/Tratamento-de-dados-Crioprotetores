import matplotlib.pyplot as plt
from pathlib import Path
from uncertainties import ufloat
from src.dados import extrair_dados_proc, extrair_tabela_calib
from processamento_dif import corrigir_anomalia, alinhar_por_temperatura, calc_erro_termopar, separar_curvas
from src.visualizacao import plot_difracao, plot_temperatura_taxa, plot_difracao_3d
from src.parametros import PARAMETROS_ANOMALIA, PERIODO_AQUISICAO_DIFRACAO, U_TEMP_CRYOJET

plt.rcParams.update(
    {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12}
)

def pipeline_completo(base_input_dir: str, output_dir: str, calib_file_path: str):
    
    input_path = Path(base_input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Leitura e aplicação de incertezas na calibração
    df_calib = extrair_tabela_calib(calib_file_path)
    
    df_calib["cryojet_current_temp[K]"] = [
        ufloat(val, U_TEMP_CRYOJET) for val in df_calib["cryojet_current_temp[K]"]
    ]
    df_calib["T1[K]"] = [
        ufloat(val, calc_erro_termopar(val)) for val in df_calib["T1[K]"]
    ]

    pastas = [f for f in input_path.iterdir() if f.is_dir()]

    for pasta in pastas:
        print(f"\nProcessando a pasta: {pasta.name}...")

        try:
            # 2. Leitura e correção de anomalias
            df_proc = extrair_dados_proc(
                str(pasta / "merge_files"), PERIODO_AQUISICAO_DIFRACAO
            )

            df_proc = corrigir_anomalia(
                PARAMETROS_ANOMALIA["targets_2theta"], 
                PARAMETROS_ANOMALIA["tol_2theta"], 
                PARAMETROS_ANOMALIA["idx_range"], 
                df_proc
            )

            df_proc["temperatura[K]"] = [
                ufloat(val, U_TEMP_CRYOJET) for val in df_proc["temperatura[K]"]
            ]

            # 3. Alinhamento e cálculo da faixa de incerteza
            df_temp_corr, stats = alinhar_por_temperatura(df_proc, df_calib)

            # 4. Construindo o dataframe final com as incertezas propagadas
            df_proc_final = df_proc.copy()
            novas_temps_u = []
            
            for t1, u_sobreposto in zip(df_temp_corr["temp_t1"], df_temp_corr["u_sobreposto"]):
                novas_temps_u.append(ufloat(t1.nominal_value, u_sobreposto))
                
            df_proc_final["temperatura[K]"] = novas_temps_u

            save_dir_path = output_path / f"{pasta.name}"
            save_dir_path.mkdir(parents=True, exist_ok=True)
            save_name = pasta.name

            # --- Plot 1: temperatura e taxas de variação x tempo
            save_plot_taxas = save_dir_path / f"taxas_temp_{save_name}_com_corr.svg"
            fig_taxas = plot_temperatura_taxa(df_proc_final, titulo=f"{save_name} (c/corr)")
            plt.savefig(save_plot_taxas, format="svg", transparent=True, bbox_inches="tight")
            plt.close(fig_taxas)

            # --- Plot 2: difratograma SEM correção
            save_plot_sem_corr = save_dir_path / f"difratograma_{save_name}_reproc_sem_corr.svg"
            fig_sem_corr = plot_difracao(df_proc, f"{save_name} (s/corr)")
            plt.savefig(save_plot_sem_corr, format="svg", transparent=True, bbox_inches="tight")
            plt.close(fig_sem_corr)

            # --- Plot 3: difratograma COM correção
            save_plot_com_corr = save_dir_path / f"difratograma_{save_name}_reproc_com_corr.svg"
            fig_com_corr = plot_difracao(df_proc_final, f"{save_name} (c/corr)")
            plt.savefig(save_plot_com_corr, format="svg", transparent=True, bbox_inches="tight")
            plt.close(fig_com_corr)

            # --- Plot 4: difratograma 3D COM correção
            df_proc_final_resf, df_proc_final_aquec, _, _ = separar_curvas(df_proc_final, "temperatura[K]")

            # - 4.1 Aquecimento
            save_plot_3d_aquec = save_dir_path / f"difratograma_3d_{save_name}_aquec.svg"
            shape_df_final_aquec = df_proc_final_aquec.shape

            fig_3d_aquec = plot_difracao_3d(
                df_proc_final_aquec, 
                f"{save_name} (aquec)",
                step_ticks_y=max(1, shape_df_final_aquec[0]//10)
            )
            plt.savefig(save_plot_3d_aquec, format="svg", transparent=True, bbox_inches="tight")
            plt.close(fig_3d_aquec)

            # - 4.2 Resfriamento
            save_plot_3d_resf = save_dir_path / f"difratograma_3d_{save_name}_resf.svg"   
            shape_df_final_resf = df_proc_final_resf.shape

            fig_3d_resf = plot_difracao_3d(
                df_proc_final_resf,
                f"{save_name} (resf)",
                step_ticks_y=max(1, shape_df_final_resf[0]//15)
            )
            plt.savefig(save_plot_3d_resf, format="svg", transparent=True, bbox_inches="tight")
            plt.close(fig_3d_resf)

            print(f" -> Plots salvos: {pasta.name}")

        except Exception as e:
            print(f"- Erro ao processar a pasta {pasta.name}: {e}")

    print("\nProcessamento concluído!")


if __name__ == "__main__":
    
    # Exemplo para dados reprocessados de julho
    pipeline_completo(
        base_input_dir="dados/02_processados/reproc_july_2025",
        output_dir="resultados/rampas/",
        calib_file_path="dados/02_processados/calibracao/tabela_calibacao_temperatura.csv",
    )

    # pipeline_completo(
    #     base_input_dir="dados/01_brutos/high_throughput",
    #     output_dir="resultados/rampas/plots_high_throughput",
    #     calib_file_path="dados/03_processados/tabela_calibacao_temperatura.csv",
    # )