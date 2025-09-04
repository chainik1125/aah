"""
Test 2x2 grid comparison - should take about 4 minutes.
"""

from aah_code.cluster_model.compare_three_methods import old_quspin_vs_new_general_heatmap
import numpy as np

if __name__ == "__main__":
    print("Testing 2x2 grid: Three-way comparison for L=20")
    print("=" * 70)
    
    # 2x2 grid = 4 points, ~4 minutes
    U_values = np.array([1.0, 3.0])
    V_values = np.array([1e-6, 1.0])  # Small non-zero value instead of 0
    
    print(f"U values: {U_values}")
    print(f"V values: {V_values}")
    print(f"Total points: {len(U_values) * len(V_values)}")
    print("\nExpected time: ~4 minutes")
    print("-" * 70)
    
    # Run with DMRG
    fig = old_quspin_vs_new_general_heatmap(
        U_values=U_values,
        V_values=V_values,
        system_size=20,
        t=1.0,
        include_dmrg=True,
        chi=32,
        int_sep_ratio=(1, 10),  # L/10 for L=20
        v_sep_ratio=(1, 2)       # π modulation
    )
    
    fig.write_html("test_2x2_grid_L20.html")
    print("\nFigure saved as 'test_2x2_grid_L20.html'")
    print("Test completed!")