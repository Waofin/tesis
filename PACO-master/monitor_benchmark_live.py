"""
Monitor en tiempo real del benchmark
Ejecuta: python monitor_benchmark_live.py
"""

import time
import json
from pathlib import Path
from datetime import datetime

output_dir = Path("output/performance_analysis/")

print("=" * 60)
print("MONITOR DE BENCHMARK EN TIEMPO REAL")
print("=" * 60)
print("Presiona Ctrl+C para detener el monitoreo\n")

last_file_time = None
iteration = 0

try:
    while True:
        iteration += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Buscar archivos JSON
        json_files = list(output_dir.glob("benchmark_results_*.json"))
        
        if json_files:
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_file = json_files[0]
            file_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
            
            # Solo mostrar si hay cambios
            if file_time != last_file_time:
                last_file_time = file_time
                
                print(f"[{current_time}] Archivo actualizado: {latest_file.name}")
                
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    
                    results = data.get('results', {})
                    
                    # Contar progreso
                    serial_count = len(results.get('serial', []))
                    threading_count = sum(len(trials) for trials in results.get('threading', {}).values())
                    loky_count = sum(len(trials) for trials in results.get('loky', {}).values())
                    
                    total_completed = serial_count + threading_count + loky_count
                    
                    print(f"  Serial: {serial_count}/3")
                    if threading_count > 0:
                        print(f"  Threading: {threading_count} ejecuciones completadas")
                    if loky_count > 0:
                        print(f"  Loky: {loky_count} ejecuciones completadas")
                    print(f"  Total: {total_completed} ejecuciones")
                    print()
                    
                except Exception as e:
                    print(f"  Error al leer: {e}\n")
        else:
            if iteration == 1:
                print(f"[{current_time}] Esperando que comience el benchmark...")
                print("  (El benchmark creara un archivo JSON cuando comience)\n")
        
        time.sleep(5)  # Verificar cada 5 segundos
        
except KeyboardInterrupt:
    print("\n\nMonitoreo detenido.")
    print("El benchmark continuara ejecutandose en segundo plano.")

