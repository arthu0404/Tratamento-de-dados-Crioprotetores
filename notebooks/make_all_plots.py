import matplotlib.pyplot as plt
from pathlib import Path
import funcoes

def faz_todos_plots(base_input_dir, output_dir, calib_file_path, periodo_aqu=23.200001):

    plt.rcParams.update(
        {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 12}
    )

    input_path = Path(base_input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_calib = funcoes.extrair_tabela_calib(calib_file_path)

    pastas = [f for f in input_path.iterdir() if f.is_dir()]

    for pasta in pastas:
        print(f"\nProcessando a pasta: {pasta.name}...")

        try:
            df_proc = funcoes.extrair_dados_proc(
                str(pasta / "merge_files"), periodo_aqu
            )

            targets_2theta_anom = [17]
            tol_2theta_anom = 0.5
            idx_range_anom = 15
            df_proc = funcoes.corrigir_anomalia(
                targets_2theta_anom, tol_2theta_anom, idx_range_anom, df_proc
            )

            df_temp_corr, stats = funcoes.alinhar_por_temperatura(df_proc, df_calib)

            df_proc_final = df_proc.copy()
            df_proc_final["temperatura[K]"] = df_temp_corr["temp_t1"].values

            temp_proc = df_proc["temperatura[K]"].values
            temp_proc_final = df_proc_final["temperatura[K]"].values
            temp_diff = temp_proc_final - temp_proc
            df_proc_diff = df_proc.copy()
            df_proc_diff["temperatura[K]"] = temp_diff

            save_dir_path = output_path / f"{pasta.name}"
            save_dir_path.mkdir(parents=True, exist_ok=True)
            save_name = pasta.name[12:]

            # --- Plot 1: temperatura e taxas de variação x tempo
            save_plot_taxas_temp_com_corr = (save_dir_path / f"taxas_temp_{save_name}_com_corr.png")
            fig_taxas_temp_com_corr = funcoes.plot_temperatura_taxa(df_proc_final, titulo=f"{save_name} (c/corr)")

            plt.savefig(save_plot_taxas_temp_com_corr, dpi=300, bbox_inches="tight")
            plt.close(fig_taxas_temp_com_corr)

            # --- Plot 2: difratograma sem correção de temperatura
            save_plot_sem_corr_path = (
                save_dir_path / f"difratograma_{save_name}_reproc_sem_corr.png"
            )
            fig_sem_corr = funcoes.plot_difracao(df_proc, f"{save_name} (s/corr)")

            plt.savefig(save_plot_sem_corr_path, dpi=300, bbox_inches="tight")
            plt.close(fig_sem_corr)

            # --- Plot 3: difratograma com correção de temperatura
            save_plot_com_corr_path = (save_dir_path / f"difratograma_{save_name}_reproc_com_corr.png")
            fig_com_corr = funcoes.plot_difracao(df_proc_final, f"{save_name} (c/corr)")

            plt.savefig(save_plot_com_corr_path, dpi=300, bbox_inches="tight")
            plt.close(fig_com_corr)

            # --- Plot 4: difratograma com as diferenças entre as temperaturas corrigidas e não corrigidas
            save_plot_diff_path = (save_dir_path / f"difratograma_{save_name}_reproc_diff.png")
            fig_diff = funcoes.plot_difracao(df_proc_diff, f"{save_name} (diff)")

            plt.savefig(save_plot_diff_path, dpi=300, bbox_inches="tight")
            plt.close(fig_diff)

            print(f" -> Plots salvos: {pasta.name}")

        except Exception as e:
            print(f"- Erro ao processar a pasta {pasta.name}: {e}")

    print("\nProcessamento concluído!")


# --- Plot dos dados reprocessados de julho de 2025 ('reproc_july_2025')
# OBS: pode ocorrer o erro [Errno 2] devio ao tamnho do caminho maior que 260 caracteres
# se isso acontecer os caminhos devem ser encurtados
faz_todos_plots(
    base_input_dir="../dados/reproc_july_2025",
    output_dir="../dados/extracted/plots_reproc_july_2025",
    calib_file_path="../dados/extracted/tabela_extraida_calib_temp.csv",
)

# --- Plot dos dados reporcessados de junho de 2024 ('reproc_jun_2024')
faz_todos_plots(
    base_input_dir="../dados/reproc_jun_2024",
    output_dir="../dados/extracted/plots_reproc_jun_2024",
    calib_file_path="../dados/extracted/tabela_extraida_calib_temp.csv",
)

