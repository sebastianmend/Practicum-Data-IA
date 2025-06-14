import os
import json
import pandas as pd

def load_all_results(results_dir="results") -> pd.DataFrame:
    """Carga y concatena todos los archivos de resultados JSON."""
    todos = []
    for nombre_archivo in os.listdir(results_dir):
        if nombre_archivo.endswith(".json"):
            ruta = os.path.join(results_dir, nombre_archivo)
            with open(ruta, "r", encoding="utf-8") as f:
                bloque = json.load(f)
                todos.extend(bloque)
    return pd.DataFrame(todos)

def export_to_excel(df: pd.DataFrame, nombre_archivo="resultados_homologacion.xlsx"):
    """Exporta el DataFrame final a Excel."""
    os.makedirs("outputs", exist_ok=True)
    ruta = os.path.join("outputs", nombre_archivo)
    df.to_excel(ruta, index=False)
    print(f"📁 Exportado exitosamente a: {ruta}")