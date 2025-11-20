"""
Script para ejecutar el benchmark extendido con más CPUs
Ejecuta: python run_extended_benchmark.py
"""

import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

# Configurar multiprocessing para Windows
import multiprocessing
multiprocessing.freeze_support()

import numpy as np
from pathlib import Path
from paco_performance_benchmark import (
    generate_test_data,
    run_benchmark,
    analyze_amdahl,
    plot_benchmark_results,
    save_results,
    BENCHMARK_CONFIG,
    analyze_scalability
)

# Configuración extendida con más CPUs
config = BENCHMARK_CONFIG.copy()
config['n_trials'] = 3
config['cpu_counts'] = [1, 2, 4, 8, 16]  # Incluyendo 16 CPUs

print("=" * 60)
print("BENCHMARK EXTENDIDO CON MÁS CPUs")
print("=" * 60)
print(f"\nConfiguración:")
for key, value in config.items():
    print(f"  {key}: {value}")

# Crear directorio de salida
Path(config['output_dir']).mkdir(parents=True, exist_ok=True)

# Generar datos de prueba
print("\nGenerando datos de prueba...")
test_data = generate_test_data(
    config['nFrames'],
    config['image_size'],
    config['angles_range'],
    seed=42
)
image_stack, angles, psf_template = test_data
print(f"  Stack shape: {image_stack.shape}")
print(f"  Ángulos shape: {angles.shape}")
print(f"  PSF template shape: {psf_template.shape}")

# Ejecutar benchmark
print("\n" + "=" * 60)
print("INICIANDO BENCHMARK EXTENDIDO")
print("=" * 60)
print("\nNOTA: Este benchmark puede tomar bastante tiempo")
print("   especialmente con 16 CPUs. Sé paciente...\n")

benchmark_results = run_benchmark(
    config, 
    test_data, 
    n_trials=config['n_trials']
)

# Analizar resultados
print("\n" + "=" * 60)
print("ANALIZANDO RESULTADOS")
print("=" * 60)
analysis = analyze_amdahl(benchmark_results)

# Análisis de escalabilidad
print("\n" + "=" * 60)
print("ANÁLISIS DE ESCALABILIDAD")
print("=" * 60)
analyze_scalability(benchmark_results, analysis)

# Visualizar
print("\nGenerando visualizaciones...")
plot_benchmark_results(
    benchmark_results, 
    analysis, 
    save_path=config['output_dir'] + 'benchmark_plots_extended.png'
)

# Guardar resultados
print("\nGuardando resultados...")
json_path, csv_path = save_results(
    benchmark_results, 
    analysis, 
    config, 
    config['output_dir']
)

print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"\nTiempo serial promedio: {analysis['serial']['mean_time']:.2f} ± {analysis['serial']['std_time']:.2f} seg")

if analysis.get('speedup_analysis', {}).get('p_mean'):
    p_mean = analysis['speedup_analysis']['p_mean']
    p_std = analysis['speedup_analysis']['p_std']
    max_theoretical = 1 / (1 - p_mean)
    print(f"\nFracción paralelizable estimada (p): {p_mean:.3f} ± {p_std:.3f}")
    print(f"Speedup máximo teórico (Amdahl): {max_theoretical:.2f}x")
    print("\nSpeedup real por método:")
    
    print("\n  Multiprocessing:")
    for n_cpu in sorted(analysis['multiprocessing'].keys()):
        if analysis['multiprocessing'][n_cpu]:
            speedup = analysis['multiprocessing'][n_cpu]['speedup']
            efficiency = analysis['multiprocessing'][n_cpu]['efficiency']
            print(f"    {n_cpu} CPUs: {speedup:.2f}x (eficiencia: {efficiency:.2%})")
    
    if analysis.get('loky'):
        print("\n  Loky:")
        for n_cpu in sorted(analysis['loky'].keys()):
            if analysis['loky'][n_cpu]:
                speedup = analysis['loky'][n_cpu]['speedup']
                efficiency = analysis['loky'][n_cpu]['efficiency']
                print(f"    {n_cpu} CPUs: {speedup:.2f}x (eficiencia: {efficiency:.2%})")

            print(f"\n[OK] Resultados guardados:")
            print(f"  JSON: {json_path}")
            print(f"  CSV: {csv_path}")
            print(f"  Graficos: {config['output_dir']}benchmark_plots_extended.png")

print("\n" + "=" * 60)
print("BENCHMARK COMPLETADO")
print("=" * 60)

