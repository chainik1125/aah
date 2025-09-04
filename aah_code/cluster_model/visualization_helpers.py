#!/usr/bin/env python3
"""
Visualization helpers for cluster model V coupling terms
Creates interactive visualizations showing supercluster structure, 
interaction patterns, and hopping coefficients.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple, Optional
import colorsys

def generate_distinct_colors(n: int) -> List[str]:
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append(f'rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})')
    return colors

def visualize_quspin_couplings_1d_chain(
    to_quspin_spinful: List,
    k_sites_supercluster: Optional[np.ndarray] = None,
    output_file: str = 'v_couplings_visualization.html',
    title: str = 'QuSpin V Coupling Visualization',
    eps: float = 1e-6,
    to_quspin_spinless: Optional[List] = None
):
    """
    Enhanced 1D chain visualization of QuSpin spinful coupling terms
    
    Creates an HTML visualization with:
    - 1D chain with curved hopping arrows and cluster highlighting
    - Labeled heatmap
    - Detailed coupling tables
    - Summary statistics
    
    Parameters
    ----------
    to_quspin_spinful : list
        The 'to_quspin_spinful' output from compute_V_couplings_bruteforce
    k_sites_supercluster : np.ndarray, optional
        Shape (num_clusters, Nc) - the supercluster structure showing which sites belong to which cluster
    output_file : str
        Output HTML filename
    title : str
        Title for the visualization
    eps : float
        Threshold for filtering out near-zero couplings (default 1e-6)
    to_quspin_spinless : list, optional
        The 'to_quspin_spinless' output for building the actual Hamiltonian matrix
    """
    
    # Check if spin sectors are identical
    up_couplings = {}
    down_couplings = {}
    
    for op_str, couplings in to_quspin_spinful:
        if op_str in ["+-|", "-+|"]:  # spin up
            up_couplings[op_str] = couplings
        elif op_str in ["|+-", "|-+"]:  # spin down
            down_couplings[op_str] = couplings
    
    # Check if spin sectors are identical
    spin_trivial = False
    if up_couplings and down_couplings:
        up_set = set((tuple(c) if isinstance(c, list) else c for c in up_couplings.get("+-|", [])))
        down_set = set((tuple(c) if isinstance(c, list) else c for c in down_couplings.get("|+-", [])))
        
        if up_set == down_set:
            spin_trivial = True
            print("Note: Spin sectors are identical. Visualizing only spin-up couplings.")
    
    # Parse all couplings and filter by epsilon
    all_couplings = []
    for op_str, couplings in to_quspin_spinful:
        for coupling in couplings:
            if len(coupling) >= 3:
                coeff, i, j = coupling[0], coupling[1], coupling[2]
                mag = float(np.abs(coeff))
                if mag >= eps:  # Only keep couplings above threshold
                    all_couplings.append({
                        'op': op_str,
                        'i': i,
                        'j': j,
                        'coeff': coeff,
                        'real': float(np.real(coeff)),
                        'imag': float(np.imag(coeff)),
                        'mag': mag
                    })
    
    if not all_couplings:
        print("No coupling pairs found in QuSpin data")
        return None
    
    # Determine system size
    max_index = max(max(c['i'], c['j']) for c in all_couplings)
    system_size = max_index + 1
    
    # Filter to spin-up if trivial
    if spin_trivial:
        viz_couplings = [c for c in all_couplings if '|' not in c['op'] or c['op'].index('|') > 0]
        spin_label = " (Spin ↑ only - spin sectors identical)"
    else:
        viz_couplings = all_couplings
        spin_label = ""
    
    # Group couplings by operator type
    couplings_by_op = {}
    for c in viz_couplings:
        if c['op'] not in couplings_by_op:
            couplings_by_op[c['op']] = []
        couplings_by_op[c['op']].append(c)
    
    # Build coupling matrix for heatmap
    coupling_matrix = np.zeros((system_size, system_size), dtype=complex)
    for c in viz_couplings:
        if c['op'] in ['+-|', '+-']:  # Only the forward terms for clarity
            coupling_matrix[c['i'], c['j']] += c['coeff']
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chain-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        svg {{
            width: 100%;
            height: auto;
            background: #fafafa;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f0f0f0;
            transition: background-color 0.3s;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 12px;
            opacity: 0.9;
            margin-top: 5px;
        }}
        .note {{
            background: #e8f4fd;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .complex {{
            color: #e74c3c;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}{spin_label}</h1>
"""
    
    # Add note if spin-trivial
    if spin_trivial:
        html += """
        <div class="note">
            <strong>Note:</strong> The spin-up and spin-down sectors have identical coupling structures. 
            Only the spin-up sector is shown below for clarity.
        </div>
"""
    
    # Statistics cards
    html += """
        <h2>Summary Statistics</h2>
        <div class="stats-grid">
"""
    
    stats = [
        ('System Size', system_size),
        ('Non-zero Couplings', len(viz_couplings)),
        ('Unique Op Types', len(couplings_by_op)),
        ('Max |Coupling|', f"{max(c['mag'] for c in viz_couplings):.6f}" if viz_couplings else "0"),
        ('Min |Coupling|', f"{min(c['mag'] for c in viz_couplings):.6f}" if viz_couplings else "0"),
        ('Threshold (eps)', f"{eps:.1e}"),
    ]
    
    for label, value in stats:
        html += f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
"""
    
    html += """
        </div>
"""
    
    # Determine cluster structure if provided
    cluster_info = {}
    cluster_colors_map = {}
    k_value_map = {}  # Map from flat_idx to actual k-value
    if k_sites_supercluster is not None:
        num_clusters, cluster_size = k_sites_supercluster.shape
        # Map flat index to cluster index AND k-value
        for cluster_idx in range(num_clusters):
            for site_idx in range(cluster_size):
                flat_idx = cluster_idx * cluster_size + site_idx
                cluster_info[flat_idx] = cluster_idx
                k_value_map[flat_idx] = k_sites_supercluster[cluster_idx, site_idx]
        
        # Generate colors for clusters
        cluster_colors_list = generate_distinct_colors(num_clusters)
        for idx, color in enumerate(cluster_colors_list):
            cluster_colors_map[idx] = color
    
    # 1D Chain Visualization
    html += """
        <h2>1D Chain with Hopping Terms</h2>
        <div class="chain-container">
"""
    
    # Create SVG for chain - position sites based on k-values
    if k_value_map:
        # Get max k-value to determine spacing
        max_k = max(k_value_map.values())
        svg_width = max(900, (max_k + 1) * 60)
    else:
        svg_width = max(900, system_size * 150)
    
    svg_height = 300
    site_y = svg_height // 2
    
    # Calculate site positions based on k-values
    if k_value_map:
        site_x_positions = {}
        x_scale = (svg_width - 200) / max(1, max(k_value_map.values()))
        start_x = 100  # Fixed left margin
        for flat_idx, k_val in k_value_map.items():
            site_x_positions[flat_idx] = start_x + k_val * x_scale
    else:
        # Fallback to sequential positioning
        site_spacing = min(120, (svg_width - 200) // max(1, system_size - 1)) if system_size > 1 else 120
        total_width = (system_size - 1) * site_spacing if system_size > 1 else 0
        start_x = (svg_width - total_width) // 2
        site_x_positions = {i: start_x + i * site_spacing for i in range(system_size)}
    
    html += f"""
        <svg viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                    <polygon points="0 0, 10 3, 0 6" fill="#e74c3c" />
                </marker>
            </defs>
"""
    
    # Add cluster legend at the top
    if cluster_info:
        legend_y = 30
        legend_x_start = 50
        for cluster_idx in range(num_clusters):
            legend_x = legend_x_start + cluster_idx * 100
            html += f"""
            <circle cx="{legend_x}" cy="{legend_y}" r="10" fill="{cluster_colors_map[cluster_idx]}" stroke="#2c3e50" stroke-width="1"/>
            <text x="{legend_x + 15}" y="{legend_y + 5}" font-size="12" fill="#333">Cluster {cluster_idx}</text>
"""
    
    # Draw chain backbone - only within clusters
    if cluster_info and system_size > 1:
        for cluster_idx in range(num_clusters):
            cluster_sites = sorted([i for i, c in cluster_info.items() if c == cluster_idx])
            # Draw lines only within each cluster
            for i in range(len(cluster_sites) - 1):
                x1 = site_x_positions[cluster_sites[i]]
                x2 = site_x_positions[cluster_sites[i+1]]
                html += f"""
            <line x1="{x1}" y1="{site_y}" x2="{x2}" y2="{site_y}" 
                  stroke="#95a5a6" stroke-width="2" stroke-dasharray="5,5"/>
"""
    elif system_size > 1:
        # Fallback if no cluster info - draw full chain
        site_positions = sorted(site_x_positions.values())
        if len(site_positions) > 1:
            html += f"""
            <line x1="{site_positions[0]}" y1="{site_y}" x2="{site_positions[-1]}" y2="{site_y}" 
                  stroke="#95a5a6" stroke-width="2" stroke-dasharray="5,5"/>
"""
    
    # Draw sites with cluster coloring and k-value labels
    for i in range(system_size):
        x = site_x_positions[i]
        # Use cluster color if available, otherwise default blue
        site_color = cluster_colors_map.get(cluster_info.get(i, -1), "#3498db")
        # Get the k-value and flat index
        k_label = k_value_map.get(i, i) if k_value_map else i
        # Show k-value as main label
        html += f"""
            <circle cx="{x}" cy="{site_y}" r="25" fill="{site_color}" stroke="#2c3e50" stroke-width="2"/>
            <text x="{x}" y="{site_y+5}" text-anchor="middle" fill="white" font-size="14" font-weight="bold">k={k_label}</text>
            <text x="{x}" y="{site_y+45}" text-anchor="middle" font-size="10" fill="#555">α={i}</text>
"""
    
    # Draw hopping arrows (curved paths)
    for c in viz_couplings:
        if c['op'] in ['+-|', '+-']:  # Only forward hops (already filtered by eps)
            i, j = c['i'], c['j']
            if i != j:  # Skip on-site terms for arrows
                x1 = site_x_positions[j]  # j is source
                x2 = site_x_positions[i]  # i is target
                
                # Calculate curve height based on distance
                distance = abs(i - j)
                curve_height = min(60, 20 + distance * 10)
                
                # Determine if curve should go up or down
                if j < i:
                    curve_y = site_y - curve_height
                else:
                    curve_y = site_y + curve_height
                
                # Create curved path
                mid_x = (x1 + x2) / 2
                
                # Adjust for arrow marker
                dx = x2 - x1
                dy = 0
                length = abs(dx)
                if length > 0:
                    x2_adjusted = x2 - (dx/length) * 25  # Stop 25px from center
                else:
                    x2_adjusted = x2
                
                path = f"M {x1},{site_y} Q {mid_x},{curve_y} {x2_adjusted},{site_y}"
                
                # Color based on magnitude
                opacity = min(1.0, c['mag'] / max(cc['mag'] for cc in viz_couplings))
                
                html += f"""
            <path d="{path}" fill="none" stroke="#e74c3c" stroke-width="2" 
                  opacity="{opacity}" marker-end="url(#arrowhead)"/>
            <text x="{mid_x}" y="{curve_y + (10 if curve_y < site_y else -10)}" 
                  text-anchor="middle" font-size="10" fill="#7f8c8d">{c['mag']:.3f}</text>
"""
    
    html += """
        </svg>
        </div>
"""
    
    # Labeled Heatmap
    html += """
        <h2>Coupling Matrix Heatmap</h2>
        <div id="heatmap"></div>
"""
    
    # Detailed coupling table
    html += """
        <h2>Detailed Coupling Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Spin</th>
                    <th>Type</th>
                    <th>Target α (i)</th>
                    <th>Source α (j)</th>
                    <th>Coefficient</th>
                    <th>Real Part</th>
                    <th>Imag Part</th>
                    <th>Magnitude</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Map operators to readable labels
    op_to_label = {
        '+-|': ('↑', 'c†c'),
        '-+|': ('↑', 'cc†'),
        '|+-': ('↓', 'c†c'),
        '|-+': ('↓', 'cc†'),
        '+-': ('spinless', 'c†c'),
        '-+': ('spinless', 'cc†')
    }
    
    for c in sorted(viz_couplings, key=lambda x: (x['op'], x['i'], x['j'])):
        # Format complex coefficient to 3dp
        if abs(c['imag']) < 1e-15:
            coeff_str = f"{c['real']:.3f}"
        elif abs(c['real']) < 1e-15:
            coeff_str = f"{c['imag']:.3f}j"
        else:
            coeff_str = f"{c['real']:.3f} {'+' if c['imag'] >= 0 else '-'} {abs(c['imag']):.3f}j"
        
        spin_label, op_type = op_to_label.get(c['op'], (c['op'], ''))
        
        html += f"""
                <tr>
                    <td>{spin_label}</td>
                    <td>{op_type}</td>
                    <td>{c['i']}</td>
                    <td>{c['j']}</td>
                    <td class="complex">{coeff_str}</td>
                    <td>{c['real']:.3f}</td>
                    <td>{c['imag']:.3f}</td>
                    <td>{c['mag']:.3f}</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
"""
    
    # Add Plotly heatmap script - fix the z data format
    html += """
    <script>
        var data = [{
            z: """ + str(np.abs(coupling_matrix).tolist()) + """,
            x: """ + str(list(range(system_size))) + """,
            y: """ + str(list(range(system_size))) + """,
            type: 'heatmap',
            colorscale: [
                [0, 'white'],
                [0.2, 'lightblue'],
                [0.5, 'blue'],
                [0.8, 'darkblue'],
                [1.0, 'black']
            ],
            showscale: true,
            zmin: 0,
            zmax: Math.max(...""" + str(np.abs(coupling_matrix).tolist()) + """.flat()),
            colorbar: {
                title: '|Coupling|',
                titleside: 'right'
            },
            hovertemplate: 'Target i: %{y}<br>Source j: %{x}<br>|Coupling|: %{z:.6f}<extra></extra>'
        }];
        
        var layout = {
            title: 'Coupling Matrix (|c_ij|)',
            xaxis: {
                title: 'Source Index (j)',
                tickmode: 'linear',
                dtick: 1,
                showgrid: true,
                gridcolor: '#e0e0e0'
            },
            yaxis: {
                title: 'Target Index (i)',
                tickmode: 'linear',
                dtick: 1,
                showgrid: true,
                gridcolor: '#e0e0e0',
                autorange: 'reversed'
            },
            height: 500,
            paper_bgcolor: '#fafafa',
            plot_bgcolor: '#ffffff'
        };
        
        Plotly.newPlot('heatmap', data, layout);
    </script>
"""
    
    # Add QuSpin Hamiltonian matrix if spinless operators provided
    if to_quspin_spinless is not None:
        try:
            from quspin.operators import hamiltonian
            from quspin.basis import spinless_fermion_basis_1d
            
            # Build the Hamiltonian for single particle
            basis = spinless_fermion_basis_1d(system_size, Nf=1)
            H = hamiltonian(to_quspin_spinless, [], basis=basis, dtype=np.complex128, check_herm=False, check_symm=False)
            H_matrix = H.toarray()
            
            # Get eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
            
            # Create basis state labels
            basis_labels = []
            for i, state in enumerate(basis.states):
                for site in range(system_size):
                    if (state >> site) & 1:
                        basis_labels.append(f"|{site}⟩")
                        break
                else:
                    basis_labels.append("|vac⟩")
            
            html += """
        <h2>QuSpin Hamiltonian (Single Particle, Spinless)</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            
            <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h3>Hamiltonian Matrix</h3>
                <table style="margin: 10px auto; font-family: monospace;">
                    <thead>
                        <tr>
                            <th style="background: #667eea; color: white; padding: 8px;">State</th>
"""
            for label in basis_labels:
                html += f'                            <th style="background: #667eea; color: white; padding: 8px;">{label}</th>\n'
            
            html += """                        </tr>
                    </thead>
                    <tbody>
"""
            
            # Add matrix rows
            for i, label_i in enumerate(basis_labels):
                html += f'                        <tr>\n'
                html += f'                            <th style="background: #667eea; color: white; padding: 8px;">{label_i}</th>\n'
                for j in range(len(basis_labels)):
                    val = H_matrix[i, j]
                    if np.abs(val) < 1e-10:
                        cell_content = "0"
                        cell_style = "padding: 8px; text-align: center; color: #999;"
                    else:
                        if np.abs(np.imag(val)) < 1e-10:
                            cell_content = f"{np.real(val):.3f}"
                        else:
                            cell_content = f"{val:.3f}"
                        cell_style = "padding: 8px; text-align: center; font-weight: bold; color: #2c3e50;"
                    html += f'                            <td style="{cell_style}">{cell_content}</td>\n'
                html += '                        </tr>\n'
            
            html += """                    </tbody>
                </table>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h3>Eigenvalues & Properties</h3>
                <table style="width: 100%; margin: 10px 0;">
                    <thead>
                        <tr>
                            <th style="background: #667eea; color: white; padding: 8px;">Index</th>
                            <th style="background: #667eea; color: white; padding: 8px;">Eigenvalue</th>
                        </tr>
                    </thead>
                    <tbody>
"""
            
            for i, eigval in enumerate(eigenvalues):
                html += f"""                        <tr>
                            <td style="padding: 8px; text-align: center;">{i}</td>
                            <td style="padding: 8px; text-align: center; font-weight: bold;">{eigval:.6f}</td>
                        </tr>
"""
            
            html += f"""                    </tbody>
                </table>
                
                <div style="margin-top: 20px; padding: 15px; background: #e8f4fd; border-radius: 5px;">
                    <h4 style="margin: 0 0 10px 0;">Properties</h4>
                    <ul style="margin: 5px 0; padding-left: 20px;">
                        <li>Dimension: {H_matrix.shape[0]} × {H_matrix.shape[1]}</li>
                        <li>Hermitian: {np.allclose(H_matrix, H_matrix.conj().T)}</li>
                        <li>Trace: {np.trace(H_matrix):.6f}</li>
                        <li>Determinant: {np.linalg.det(H_matrix):.6f}</li>
                        <li>Condition number: {np.linalg.cond(H_matrix):.3e}</li>
                    </ul>
                </div>
            </div>
        </div>
"""
            
        except ImportError:
            html += """
        <div class="note" style="background: #fff3cd; border-left: 4px solid #ffc107;">
            <strong>Note:</strong> QuSpin not available for Hamiltonian matrix display.
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Visualization saved to {output_file}")
    return True


def visualize_quspin_couplings(
    to_quspin_spinful: List,
    output_file: str = 'v_couplings_visualization.html',
    title: str = 'QuSpin V Coupling Visualization'
):
    """
    One-line visualization of QuSpin spinful coupling terms from compute_V_couplings_bruteforce
    
    Parameters
    ----------
    to_quspin_spinful : list
        The 'to_quspin_spinful' output from compute_V_couplings_bruteforce
        Format: [["+-|", hop_ij], ["-+|", hop_ji_hc], ["|+-", hop_ij], ["|-+", hop_ji_hc]]
    output_file : str
        Output HTML filename
    title : str
        Title for the visualization
    """
    
    # Check if spin sectors are identical (trivial spin structure)
    spin_up_ops = [op_str for op_str, _ in to_quspin_spinful if '|' in op_str and op_str.index('|') < len(op_str)-1]
    spin_down_ops = [op_str for op_str, _ in to_quspin_spinful if '|' in op_str and op_str.index('|') == 0]
    
    # Extract couplings for spin up
    up_couplings = {}
    down_couplings = {}
    
    for op_str, couplings in to_quspin_spinful:
        if op_str in ["+-|", "-+|"]:  # spin up
            up_couplings[op_str] = couplings
        elif op_str in ["|+-", "|-+"]:  # spin down
            down_couplings[op_str] = couplings
    
    # Check if spin sectors are identical
    spin_trivial = False
    if up_couplings and down_couplings:
        # Compare the coupling lists (ignoring operator strings)
        up_set = set((tuple(c) if isinstance(c, list) else c for c in up_couplings.get("+-|", [])))
        down_set = set((tuple(c) if isinstance(c, list) else c for c in down_couplings.get("|+-", [])))
        
        if up_set == down_set:
            spin_trivial = True
            print("Note: Spin sectors are identical. Visualizing only spin-up couplings.")
    
    # Select which couplings to visualize
    if spin_trivial:
        couplings_to_viz = up_couplings
        op_prefix = "Spin ↑: "
    else:
        couplings_to_viz = to_quspin_spinful
        op_prefix = ""
    
    # Parse coupling data
    all_pairs = []
    op_types = []
    
    if isinstance(couplings_to_viz, dict):
        # Just spin up or down
        for op_str, couplings in couplings_to_viz.items():
            for coupling in couplings:
                if len(coupling) >= 3:
                    coeff, i, j = coupling[0], coupling[1], coupling[2]
                    all_pairs.append((i, j, coeff, op_str))
                    op_types.append(op_str)
    else:
        # Full spinful list
        for op_str, couplings in couplings_to_viz:
            for coupling in couplings:
                if len(coupling) >= 3:
                    coeff, i, j = coupling[0], coupling[1], coupling[2]
                    all_pairs.append((i, j, coeff, op_str))
                    op_types.append(op_str)
    
    if not all_pairs:
        print("No coupling pairs found in QuSpin data")
        return None
    
    # Determine the size of the system
    max_index = max(max(i, j) for i, j, _, _ in all_pairs)
    system_size = max_index + 1
    
    # Create visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{op_prefix}Coupling Network Graph',
            f'{op_prefix}Coupling Matrix (|c|)',
            f'{op_prefix}Coupling Strength Distribution',
            'Coupling Details'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'heatmap'}],
            [{'type': 'histogram'}, {'type': 'table'}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # ===== 1. Network Graph =====
    # Position sites in a circle
    angles = 2 * np.pi * np.arange(system_size) / system_size
    x_pos = np.cos(angles) * 3
    y_pos = np.sin(angles) * 3
    
    # Plot sites
    fig.add_trace(
        go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            marker=dict(size=20, color='lightblue', line=dict(color='darkblue', width=2)),
            text=[str(i) for i in range(system_size)],
            textposition='middle center',
            name='Sites',
            showlegend=True,
            hovertemplate='Site %{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add coupling arrows - group by operator type
    op_colors = {'+-': 'red', '-+': 'blue', '+-|': 'red', '-+|': 'blue', 
                 '|+-': 'orange', '|-+': 'purple'}
    
    for op_str in set(op_types):
        op_pairs = [(i, j, c) for i, j, c, op in all_pairs if op == op_str]
        
        for i, j, coeff in op_pairs:
            if i != j:  # Skip self-couplings for arrows
                # Arrow from j to i (since it's c†_i c_j)
                fig.add_annotation(
                    x=x_pos[i], y=y_pos[i],
                    ax=x_pos[j], ay=y_pos[j],
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=max(0.5, min(3, np.abs(coeff) * 2)),
                    arrowcolor=op_colors.get(op_str.replace('|', ''), 'gray'),
                    opacity=0.6,
                    row=1, col=1
                )
    
    # ===== 2. Coupling Matrix =====
    # Build coupling matrix
    coupling_matrix = np.zeros((system_size, system_size), dtype=complex)
    for i, j, coeff, _ in all_pairs:
        coupling_matrix[i, j] += coeff
    
    fig.add_trace(
        go.Heatmap(
            z=np.abs(coupling_matrix),
            colorscale='Viridis',
            colorbar=dict(title='|c|'),
            hovertemplate='i=%{y}, j=%{x}<br>|c|=%{z:.6f}<extra></extra>',
        ),
        row=1, col=2
    )
    
    # ===== 3. Coupling Strength Distribution =====
    magnitudes = [np.abs(c) for _, _, c, _ in all_pairs]
    
    fig.add_trace(
        go.Histogram(
            x=magnitudes,
            nbinsx=20,
            name='|Coupling|',
            showlegend=False,
            marker_color='green',
            hovertemplate='|c|=%{x:.6f}<br>Count=%{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # ===== 4. Coupling Summary Table =====
    # Create summary statistics
    unique_ops = list(set(op_types))
    stats_data = []
    
    stats_data.append(['Total couplings', str(len(all_pairs))])
    stats_data.append(['System size', str(system_size)])
    stats_data.append(['Operator types', ', '.join(unique_ops)])
    
    if magnitudes:
        stats_data.append(['Max |coupling|', f'{np.max(magnitudes):.6f}'])
        stats_data.append(['Min |coupling|', f'{np.min(magnitudes):.6f}'])
        stats_data.append(['Mean |coupling|', f'{np.mean(magnitudes):.6f}'])
        stats_data.append(['Std |coupling|', f'{np.std(magnitudes):.6f}'])
    
    # Count self vs non-self couplings
    self_couplings = sum(1 for i, j, _, _ in all_pairs if i == j)
    hop_couplings = len(all_pairs) - self_couplings
    stats_data.append(['Self-couplings', str(self_couplings)])
    stats_data.append(['Hopping terms', str(hop_couplings)])
    
    if spin_trivial:
        stats_data.append(['Spin structure', 'Trivial (identical ↑/↓)'])
    else:
        stats_data.append(['Spin structure', 'Non-trivial'])
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Property', 'Value'],
                fill_color='lightgrey',
                align='left'
            ),
            cells=dict(
                values=list(zip(*stats_data)),
                align='left'
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=title,
        showlegend=True,
        height=900,
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_yaxes(title_text='y', row=1, col=1)
    fig.update_xaxes(title_text='j (source)', row=1, col=2)
    fig.update_yaxes(title_text='i (target)', row=1, col=2)
    fig.update_xaxes(title_text='|Coupling|', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)
    
    # Equal aspect ratio for network graph
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    
    # Save to file
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")
    
    return fig


def visualize_v_couplings(
    res: Dict,
    k_sites_supercluster: np.ndarray,
    V_separation: int,
    L: int,
    output_file: str = 'v_couplings_visualization.html'
):
    """
    Create interactive visualization of V coupling terms from compute_V_couplings_bruteforce
    
    Creates a 4-panel visualization showing:
    1. Supercluster structure with k-site labels and cluster grouping
    2. Alpha-basis coupling matrix heatmap
    3. Interaction flow diagram between clusters
    4. Summary statistics table
    
    Parameters
    ----------
    res : dict
        Output from compute_V_couplings_bruteforce containing:
        - pairs: list of (i, j, coeff) tuples
        - blocks: dict of coupling blocks by (mu', mu) keys
        - validation: optional validation results
    k_sites_supercluster : np.ndarray
        Shape (num_clusters, Nc) - k-site labels for supercluster
    V_separation : int
        The V separation parameter (n in k -> k+n)
    L : int
        Total number of k-points
    output_file : str
        Output HTML filename
    """
    
    num_clusters, Nc = k_sites_supercluster.shape
    total_sites = num_clusters * Nc
    
    # Extract data from results
    pairs = res.get('pairs', [])
    blocks = res.get('blocks', {})
    validation = res.get('validation', {})
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Supercluster Structure',
            'Alpha-Basis Coupling Matrix',
            'Interaction Flow Diagram',
            'Summary Statistics'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'heatmap'}],
            [{'type': 'scatter'}, {'type': 'table'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # Generate cluster colors
    cluster_colors = generate_distinct_colors(num_clusters)
    
    # ===== 1. Supercluster Structure =====
    # Position clusters in a circle for better visualization
    cluster_positions = []
    for mu in range(num_clusters):
        angle = 2 * np.pi * mu / num_clusters
        x_base = 3 * np.cos(angle)
        y_base = 3 * np.sin(angle)
        cluster_positions.append((x_base, y_base))
    
    # Plot k-sites with cluster grouping
    for mu in range(num_clusters):
        x_base, y_base = cluster_positions[mu]
        
        # Position sites within cluster
        for a in range(Nc):
            x = x_base + 0.3 * np.cos(2 * np.pi * a / Nc)
            y = y_base + 0.3 * np.sin(2 * np.pi * a / Nc)
            k = k_sites_supercluster[mu, a]
            
            # Calculate where this k-site hops to
            k_prime = (k + V_separation) % L
            
            fig.add_trace(
                go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=20, color=cluster_colors[mu]),
                    text=f'k={k}<br>μ={mu},a={a}',
                    textposition='top center',
                    name=f'Cluster {mu}',
                    showlegend=(a == 0),
                    hovertemplate=f'k={k}<br>Cluster μ={mu}<br>Index a={a}<br>k→k+{V_separation}={k_prime}',
                ),
                row=1, col=1
            )
    
    # Add V-separation arrows
    for mu in range(num_clusters):
        for a in range(Nc):
            k = k_sites_supercluster[mu, a]
            k_prime = (k + V_separation) % L
            
            # Find target position if in supercluster
            target_found = False
            for mu_p in range(num_clusters):
                for b in range(Nc):
                    if k_sites_supercluster[mu_p, b] == k_prime:
                        x1_base, y1_base = cluster_positions[mu]
                        x1 = x1_base + 0.3 * np.cos(2 * np.pi * a / Nc)
                        y1 = y1_base + 0.3 * np.sin(2 * np.pi * a / Nc)
                        
                        x2_base, y2_base = cluster_positions[mu_p]
                        x2 = x2_base + 0.3 * np.cos(2 * np.pi * b / Nc)
                        y2 = y2_base + 0.3 * np.sin(2 * np.pi * b / Nc)
                        
                        fig.add_annotation(
                            x=x2, y=y2,
                            ax=x1, ay=y1,
                            xref="x", yref="y",
                            axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor='rgba(100,100,100,0.5)',
                            row=1, col=1
                        )
                        target_found = True
                        break
                if target_found:
                    break
    
    # ===== 2. Alpha-Basis Coupling Matrix =====
    # Create coupling matrix from pairs
    coupling_matrix = np.zeros((total_sites, total_sites), dtype=complex)
    for i, j, c in pairs:
        coupling_matrix[i, j] = c
    
    # Plot magnitude of couplings
    fig.add_trace(
        go.Heatmap(
            z=np.abs(coupling_matrix),
            colorscale='Viridis',
            colorbar=dict(title='|Coupling|'),
            hovertemplate='α=%{y}, β=%{x}<br>|c|=%{z}<extra></extra>',
        ),
        row=1, col=2
    )
    
    # Add cluster boundaries
    for mu in range(1, num_clusters):
        boundary = mu * Nc - 0.5
        fig.add_shape(
            type="line",
            x0=boundary, y0=-0.5,
            x1=boundary, y1=total_sites-0.5,
            line=dict(color="white", width=1, dash="dash"),
            row=1, col=2
        )
        fig.add_shape(
            type="line",
            x0=-0.5, y0=boundary,
            x1=total_sites-0.5, y1=boundary,
            line=dict(color="white", width=1, dash="dash"),
            row=1, col=2
        )
    
    # ===== 3. Interaction Flow Diagram =====
    # Create a simplified flow diagram showing block structure
    for (mu_p, mu), T in blocks.items():
        x1, y1 = cluster_positions[mu]
        x2, y2 = cluster_positions[mu_p]
        
        # Calculate coupling strength
        strength = np.linalg.norm(T, 'fro')
        
        if strength > 1e-10:  # Only show significant couplings
            # Add arrow
            fig.add_annotation(
                x=x2, y=y2,
                ax=x1, ay=y1,
                xref="x3", yref="y3",
                axref="x3", ayref="y3",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=np.log1p(strength) + 1,
                arrowcolor='red',
                opacity=min(1.0, strength/np.max([np.linalg.norm(b, 'fro') for b in blocks.values()])),
                row=2, col=1
            )
            
            # Add label at midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            fig.add_trace(
                go.Scatter(
                    x=[mid_x], y=[mid_y],
                    mode='text',
                    text=[f'{strength:.3f}'],
                    textposition='middle center',
                    showlegend=False,
                    hovertemplate=f'μ={mu}→μ\'={mu_p}<br>Strength={strength:.4f}'
                ),
                row=2, col=1
            )
    
    # Add cluster nodes
    for mu in range(num_clusters):
        x, y = cluster_positions[mu]
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=40, color=cluster_colors[mu]),
                text=f'μ={mu}',
                textposition='middle center',
                showlegend=False,
                hovertemplate=f'Cluster μ={mu}'
            ),
            row=2, col=1
        )
    
    # ===== 4. Summary Statistics Table =====
    # Collect statistics
    stats_data = []
    
    # Basic info
    stats_data.append(['Total k-sites', str(total_sites)])
    stats_data.append(['Number of clusters', str(num_clusters)])
    stats_data.append(['Cluster size (Nc)', str(Nc)])
    stats_data.append(['V separation', f'{V_separation} (mod {L})'])
    stats_data.append(['Total couplings', str(len(pairs))])
    
    # Coupling statistics
    if pairs:
        coeffs = [np.abs(c) for _, _, c in pairs]
        stats_data.append(['Max |coupling|', f'{np.max(coeffs):.6f}'])
        stats_data.append(['Min |coupling|', f'{np.min(coeffs):.6f}'])
        stats_data.append(['Mean |coupling|', f'{np.mean(coeffs):.6f}'])
    
    # Block structure
    stats_data.append(['Number of blocks', str(len(blocks))])
    within_block = sum(1 for (mu_p, mu) in blocks.keys() if mu_p == mu)
    between_block = len(blocks) - within_block
    stats_data.append(['Within-cluster blocks', str(within_block)])
    stats_data.append(['Between-cluster blocks', str(between_block)])
    
    # Validation results if available
    if validation:
        stats_data.append(['Validation OK', str(validation.get('ok', 'N/A'))])
        stats_data.append(['Max validation error', f"{validation.get('max_abs_err', 0):.2e}"])
    
    # Create table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Property', 'Value'],
                fill_color='lightgrey',
                align='left'
            ),
            cells=dict(
                values=list(zip(*stats_data)),
                align='left'
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=f'V Coupling Visualization (V_sep={V_separation}, L={L})',
        showlegend=True,
        height=900,
        hovermode='closest'
    )
    
    # Update axes
    fig.update_xaxes(title_text='x', row=1, col=1)
    fig.update_yaxes(title_text='y', row=1, col=1)
    fig.update_xaxes(title_text='β (source)', row=1, col=2)
    fig.update_yaxes(title_text='α (target)', row=1, col=2)
    fig.update_xaxes(title_text='x', row=2, col=1)
    fig.update_yaxes(title_text='y', row=2, col=1)
    
    # Equal aspect ratio for scatter plots
    fig.update_xaxes(scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_xaxes(scaleanchor="y3", scaleratio=1, row=2, col=1)
    
    # Write to HTML file
    fig.write_html(output_file)
    print(f"Visualization saved to {output_file}")
    
    return fig


def create_coupling_details_table(res: Dict, output_file: str = 'coupling_details.html'):
    """
    Create a detailed HTML table of all coupling terms
    
    Parameters
    ----------
    res : dict
        Output from compute_V_couplings_bruteforce containing pairs
    output_file : str
        Output HTML filename
    
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the coupling details
    """
    pairs = res.get('pairs', [])
    
    if not pairs:
        print("No coupling pairs found")
        return None
    
    # Convert to DataFrame for easier manipulation
    data = []
    for i, j, c in pairs:
        data.append({
            'Target (i)': i,
            'Source (j)': j,
            'Real Part': np.real(c),
            'Imag Part': np.imag(c),
            'Magnitude': np.abs(c),
            'Phase (deg)': np.angle(c, deg=True)
        })
    
    df = pd.DataFrame(data)
    
    # Create HTML table with styling
    html = df.to_html(index=False, float_format=lambda x: f'{x:.6f}')
    
    # Add CSS styling
    styled_html = f"""
    <html>
    <head>
        <title>V Coupling Details</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #ddd; }}
        </style>
    </head>
    <body>
        <h1>V Coupling Terms - Detailed Table</h1>
        {html}
        <p>Total number of couplings: {len(pairs)}</p>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(styled_html)
    
    print(f"Detailed coupling table saved to {output_file}")
    return df