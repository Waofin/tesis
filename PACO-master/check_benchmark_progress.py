"""
Script para verificar el progreso del benchmark
Ejecuta: python check_benchmark_progress.py
"""

import json
from pathlib import Path
from datetime import datetime

output_dir = Path("output/performance_analysis/")

print("=" * 60)
print("ESTADO DEL BENCHMARK")
print("=" * 60)

# Buscar archivos JSON de resultados
json_files = list(output_dir.glob("benchmark_results_*.json"))

if json_files:
    # Ordenar por fecha de modificación (más reciente primero)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = json_files[0]
    
    print(f"\nArchivo mas reciente: {latest_file.name}")
    print(f"Modificado: {datetime.fromtimestamp(latest_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Leer y mostrar resumen
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', {})
        config = data.get('config', {})
        
        print(f"\nConfiguracion:")
        print(f"  CPUs probados: {config.get('cpu_counts', [])}")
        print(f"  Trials: {config.get('n_trials', 'N/A')}")
        
        print(f"\nResultados completados:")
        
        # Serial
        if results.get('serial'):
            serial_times = [r['time'] for r in results['serial']]
            print(f"  Serial: {len(serial_times)}/{config.get('n_trials', 0)} trials completados")
            if serial_times:
                print(f"    Tiempo promedio: {sum(serial_times)/len(serial_times):.2f} seg")
        
        # Threading
        if results.get('threading'):
            threading_completed = {}
            for n_cpu, trials in results['threading'].items():
                if trials:
                    threading_completed[n_cpu] = len(trials)
                    times = [r['time'] for r in trials]
                    avg_time = sum(times)/len(times)
                    print(f"  Threading ({n_cpu} CPUs): {len(trials)}/{config.get('n_trials', 0)} trials - Tiempo promedio: {avg_time:.2f} seg")
        
        # Loky
        if results.get('loky'):
            loky_completed = {}
            for n_cpu, trials in results['loky'].items():
                if trials:
                    loky_completed[n_cpu] = len(trials)
                    times = [r['time'] for r in trials]
                    avg_time = sum(times)/len(times)
                    print(f"  Loky ({n_cpu} CPUs): {len(trials)}/{config.get('n_trials', 0)} trials - Tiempo promedio: {avg_time:.2f} seg")
        
        # Calcular progreso total
        total_trials = config.get('n_trials', 0)
        cpu_counts = config.get('cpu_counts', [])
        
        # Serial: 1 configuracion
        # Threading: len(cpu_counts) - 1 (excluyendo 1 CPU)
        # Loky: len(cpu_counts) - 1 (excluyendo 1 CPU)
        total_configs = 1 + 2 * (len(cpu_counts) - 1)
        total_expected = total_configs * total_trials
        
        completed = 0
        if results.get('serial'):
            completed += len(results['serial'])
        if results.get('threading'):
            for trials in results['threading'].values():
                completed += len(trials)
        if results.get('loky'):
            for trials in results['loky'].values():
                completed += len(trials)
        
        progress = (completed / total_expected * 100) if total_expected > 0 else 0
        print(f"\nProgreso total: {completed}/{total_expected} ejecuciones ({progress:.1f}%)")
        
        if completed == total_expected:
            print("\n[OK] Benchmark completado!")
        else:
            print("\n[EN PROGRESO] El benchmark aun esta ejecutandose...")
        
    except Exception as e:
        print(f"\nError al leer archivo: {e}")
else:
    print("\nNo se encontraron archivos de resultados aun.")
    print("El benchmark puede estar ejecutandose o no ha comenzado.")

print("\n" + "=" * 60)

