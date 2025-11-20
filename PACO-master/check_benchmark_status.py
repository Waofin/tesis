"""
Script para verificar el estado del benchmark
Ejecuta: python check_benchmark_status.py
"""

import os
from pathlib import Path
import json
from datetime import datetime

output_dir = Path('output/performance_analysis/')

print("=" * 60)
print("ESTADO DEL BENCHMARK")
print("=" * 60)

# Verificar si hay archivos de resultados
json_files = list(output_dir.glob('benchmark_results_*.json'))
csv_files = list(output_dir.glob('benchmark_summary_*.csv'))
png_files = list(output_dir.glob('benchmark_plots*.png'))

if json_files:
    # Obtener el archivo más reciente
    latest_json = max(json_files, key=os.path.getmtime)
    latest_csv = max(csv_files, key=os.path.getmtime) if csv_files else None
    
    print(f"\n[OK] Archivos de resultados encontrados:")
    print(f"   JSON mas reciente: {latest_json.name}")
    if latest_csv:
        print(f"   CSV mas reciente: {latest_csv.name}")
    
    # Leer y mostrar resumen
    try:
        with open(latest_json, 'r') as f:
            data = json.load(f)
        
        config = data.get('config', {})
        results = data.get('results', {})
        analysis = data.get('analysis', {})
        
        print(f"\n[INFO] Configuracion del benchmark:")
        print(f"   CPUs probados: {config.get('cpu_counts', 'N/A')}")
        print(f"   Numero de trials: {config.get('n_trials', 'N/A')}")
        
        if analysis.get('serial'):
            serial_time = analysis['serial']['mean_time']
            print(f"\n[TIME] Tiempo serial promedio: {serial_time:.2f} segundos")
        
        if analysis.get('speedup_analysis', {}).get('p_mean'):
            p_mean = analysis['speedup_analysis']['p_mean']
            max_theoretical = 1 / (1 - p_mean)
            print(f"\n[AMDahl] Analisis de Amdahl:")
            print(f"   Fraccion paralelizable (p): {p_mean:.3f}")
            print(f"   Speedup maximo teorico: {max_theoretical:.2f}x")
        
        # Verificar si hay resultados para 16 CPUs
        mp_results = analysis.get('multiprocessing', {})
        loky_results = analysis.get('loky', {})
        
        has_16_cpus = False
        if 16 in mp_results and mp_results[16]:
            has_16_cpus = True
            print(f"\n[OK] Benchmark extendido completado (incluye 16 CPUs)")
        elif 16 in loky_results and loky_results[16]:
            has_16_cpus = True
            print(f"\n[OK] Benchmark extendido completado (incluye 16 CPUs)")
        else:
            print(f"\n[WAIT] Benchmark basico completado (hasta 8 CPUs)")
            print(f"   El benchmark extendido puede estar aun ejecutandose...")
        
    except Exception as e:
        print(f"\n[ERROR] Error al leer resultados: {e}")
else:
    print(f"\n[WAIT] No se encontraron archivos de resultados aun.")
    print(f"   El benchmark puede estar ejecutandose...")
    print(f"\n[TIP] Para verificar procesos Python activos:")
    print(f"   En PowerShell: Get-Process python")
    print(f"   En CMD: tasklist | findstr python")

print("\n" + "=" * 60)
print("Para ejecutar el benchmark extendido:")
print("  python run_extended_benchmark.py")
print("=" * 60)
