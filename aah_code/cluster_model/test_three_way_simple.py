"""
Simple three-way comparison test with V=0 only.
"""

from aah_code.cluster_model.compare_three_methods import old_quspin_vs_new_general_heatmap
import numpy as np

if __name__ == "__main__":
    print("Quick test: Three-way comparison for L=20 with V=0")
    print("=" * 70)
    
    # Just test with V=0 for speed
    U_values = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    V_values = np.array([0.0])  # Just V=0
    
    # Run comparison without DMRG first (faster)
    print("\n1. Testing without DMRG (faster)...")
    fig_no_dmrg = old_quspin_vs_new_general_heatmap(
        U_values=U_values,
        V_values=V_values,
        system_size=20,
        t=1.0,
        include_dmrg=False,
        int_sep_ratio=(1, 10),
        v_sep_ratio=(1, 2)
    )
    
    fig_no_dmrg.write_html("quick_test_L20_no_dmrg.html")
    print("Figure saved as 'quick_test_L20_no_dmrg.html'")
    
    # Now with DMRG
    print("\n2. Testing with DMRG...")
    fig = old_quspin_vs_new_general_heatmap(
        U_values=U_values,
        V_values=V_values,
        system_size=20,
        t=1.0,
        include_dmrg=True,
        chi=32,
        int_sep_ratio=(1, 10),
        v_sep_ratio=(1, 2)
    )
    
    fig.write_html("quick_test_L20_with_dmrg.html")
    print("Figure saved as 'quick_test_L20_with_dmrg.html'")
    
    print("\nTest completed!")