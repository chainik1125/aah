#!/bin/bash
# RunPod Results Download Script
# Generated: 2025-09-11 17:13:33.413321

echo "Downloading RunPod results..."
mkdir -p runpod_results

echo "- int_sep_comparison_v_sep_1_2_U_0p0_Nc2_page_1.html"
# scp command: scp user@host:/root/aah/aah_code/cluster_model/large_files/plots/int_sep_comparison_v_sep_1_2_U_0p0_Nc2_page_1.html runpod_results/
echo "- int_sep_comparison_v_sep_1_2_U_0p0_Nc2_page_2.html"
# scp command: scp user@host:/root/aah/aah_code/cluster_model/large_files/plots/int_sep_comparison_v_sep_1_2_U_0p0_Nc2_page_2.html runpod_results/

echo "Done! Files are in runpod_results/"
echo ""
echo "To download, use:"
echo "  scp -P [PORT] root@[RUNPOD_IP]:/root/aah/large_files/plots/*.html ./runpod_results/"
