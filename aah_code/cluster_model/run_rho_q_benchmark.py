import numpy as np

from aah_code.cluster_model.plots import compare_U_values_rhoQ_with_dmrg


def main():
    v_sep_ratio = (1, 2)
    int_sep_ratios = {
        2: (1, 2),
        4: (1, 4),
    }
    cluster_sizes = [2, 4]
    U_values = np.linspace(0, 4, 5)
    V_values = np.array([0.0, 1.0, 2.0])

    compare_U_values_rhoQ_with_dmrg(
        v_sep_ratio=v_sep_ratio,
        int_sep_ratios=int_sep_ratios,
        cluster_sizes=cluster_sizes,
        U_values=U_values,
        V_values=V_values,
        t=1.0,
        L=20,
        chi=32,
        solver_method='sparse_ED',
        states_retained=4,
        output_dir='large_files/plots',
        show_plots=True,
        save_html=True,
        save_data=True,
        filename_prefix='paper_rho_q_comparison',
    )


if __name__ == "__main__":
    main()
