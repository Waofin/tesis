"""
Script para monitorear el progreso del benchmark en tiempo real
Ejecuta: python monitor_benchmark.py
"""

import os
import time
from pathlib import Path
import json

output_dir = Path('output/performance_analysis/')

print("=" * 60)
print("MONITOR DE BENCHMARK")
print("=" * 60)
print("\nPresiona Ctrl+C para detener el monitoreo\n")

last_file_count = 0
last_json_time = 0

try:
    while True:
        # Verificar archivos de resultados
        json_files = list(output_dir.glob('benchmark_results_*.json'))
        csv_files = list(output_dir.glob('benchmark_summary_*.csv'))
        
        current_count = len(json_files)
        
        if json_files:
            latest_json = max(json_files, key=os.path.getmtime)
            current_time = os.path.getmtime(latest_json)
            
            # Si hay un archivo nuevo o modificado
            if current_count > last_file_count or current_time > last_json_time:
                print(f"\n[{time.strftime('%H:%M:%S')}] Archivo actualizado: {latest_json.name}")
                
                try:
                    with open(latest_json, 'r') as f:
                        data = json.load(f)
                    
                    config = data.get('config', {})
                    results = data.get('results', {})
                    
                    cpu_counts = config.get('cpu_counts', [])
                    n_trials = config.get('n_trials', 0)
                    
                    # Contar trials completados
                    serial_trials = len(results.get('serial', []))
                    mp_trials = {}
                    loky_trials = {}
                    
                    for n_cpu in cpu_counts:
                        if n_cpu > 1:
                            mp_data = results.get('multiprocessing', {}).get(n_cpu, [])
                            loky_data = results.get('loky', {}).get(n_cpu, [])
                            mp_trials[n_cpu] = len(mp_data)
                            loky_trials[n_cpu] = len(loky_data)
                    
                    print(f"   Serial: {serial_trials}/{n_trials} trials completados")
                    print(f"   CPUs probados: {cpu_counts}")
                    
                    for n_cpu in sorted(mp_trials.keys()):
                        mp_count = mp_trials[n_cpu]
                        loky_count = loky_trials.get(n_cpu, 0)
                        print(f"   {n_cpu} CPUs - MP: {mp_count}/{n_trials}, Loky: {loky_count}/{n_trials}")
                    
                    # Verificar si está completo
                    total_expected = n_trials * (1 + 2 * (len(cpu_counts) - 1))  # serial + (mp + loky) para cada CPU > 1
                    total_completed = serial_trials + sum(mp_trials.values()) + sum(loky_trials.values())
                    
                    if total_completed >= total_expected:
                        print(f"\n[COMPLETO] Benchmark finalizado!")
                        break
                    else:
                        progress = (total_completed / total_expected) * 100
                        print(f"   Progreso: {progress:.1f}% ({total_completed}/{total_expected} trials)")
                
                except Exception as e:
                    print(f"   Error al leer archivo: {e}")
                
                last_file_count = current_count
                last_json_time = current_time
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Esperando actualizaciones...", end='\r')
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Esperando inicio del benchmark...", end='\r')
        
        time.sleep(5)  # Verificar cada 5 segundos

except KeyboardInterrupt:
    print("\n\nMonitoreo detenido por el usuario.")
    print("El benchmark puede seguir ejecutandose en segundo plano.")

