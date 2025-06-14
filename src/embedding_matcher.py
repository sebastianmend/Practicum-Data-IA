# embedding_matcher.py - versión estable sin errores de 'embedding'

import os
import pandas as pd
import torch
import pickle
import json
from typing import Dict
from unidecode import unidecode
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm

# === CONFIGURACIÓN GENERAL ===
HUGGINGFACE_MODEL = "tiiuae/falcon-rw-1b"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
SIM_THRESHOLD = 0.25
RESULTS_DIR = "results"
DESC_CACHE_PATH = "models/descripcion_cache.pkl"
EMB_CACHE_PATH = "models/embedding_cache.pkl"

# === CARGAR ENTORNOS Y MODELOS ===
load_dotenv()
login(token=os.getenv("HUGGINGFACE_TOKEN"))

model = SentenceTransformer(EMBEDDING_MODEL, device='cuda' if torch.cuda.is_available() else 'cpu')
generator = pipeline("text-generation", model=HUGGINGFACE_MODEL, device=-1)

# === CACHES ===
def load_cache(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def save_cache(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)

DESC_CACHE = load_cache(DESC_CACHE_PATH)
EMB_CACHE = load_cache(EMB_CACHE_PATH)

# === UTILS ===
def clean_text(text):
    return unidecode(text.strip().lower()) if isinstance(text, str) else ""

def generate_description(carrera, campo, nivel):
    clave = f"{carrera}||{campo}||{nivel}"
    if clave in DESC_CACHE:
        return DESC_CACHE[clave]
    try:
        prompt = f"Carrera: {carrera}\nCampo: {campo}\nNivel: {nivel}\nDescribe las materias clave, competencias principales y enfoque académico."
        result = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
        descripcion = result[0]['generated_text'].strip()
    except:
        descripcion = f"{nivel.title()} en {carrera.title()} del campo {campo.title()}, con formación en materias clave del área."
    DESC_CACHE[clave] = descripcion
    return descripcion

def get_embedding(descripcion):
    if descripcion in EMB_CACHE:
        return EMB_CACHE[descripcion]
    emb = model.encode(descripcion, convert_to_tensor=True).cpu()
    EMB_CACHE[descripcion] = emb
    return emb

def prepare_data(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['nombre_ies'] = df['nombre_ies'].apply(clean_text)
    df['nombre_carrera'] = df['nombre_carrera'].apply(clean_text)
    df['campo_amplio'] = df['campo_amplio'].apply(clean_text)
    df['nivel_formacion'] = df['nivel_formacion'].apply(clean_text)
    df['id'] = df.index
    df['clave'] = df.apply(lambda r: f"{r['nombre_carrera']}||{r['campo_amplio']}||{r['nivel_formacion']}", axis=1)
    return df

def homologar_optimizado(df):
    df = prepare_data(df)
    df_utpl = df[df['nombre_ies'] == 'universidad tecnica particular de loja'].copy()
    df_ext = df[df['nombre_ies'] != 'universidad tecnica particular de loja'].copy()

    claves_unicas = pd.concat([df_utpl, df_ext])['clave'].unique()
    print(f"🔍 Generando descripciones y embeddings para {len(claves_unicas)} claves únicas...")

    for clave in tqdm(claves_unicas, desc="🧠 Preprocesando descripciones"):
        if clave not in DESC_CACHE:
            carrera, campo, nivel = clave.split("||")
            _ = generate_description(carrera, campo, nivel)

    save_cache(DESC_CACHE, DESC_CACHE_PATH)

    descripciones = {clave: DESC_CACHE[clave] for clave in claves_unicas}

    for desc in tqdm(descripciones.values(), desc="🔗 Generando embeddings"):
        if desc not in EMB_CACHE:
            EMB_CACHE[desc] = model.encode(desc, convert_to_tensor=True).cpu()

    save_cache(EMB_CACHE, EMB_CACHE_PATH)

    # Asignar descripción y embedding asegurando existencia
    df['descripcion'] = df['clave'].map(DESC_CACHE)
    df['embedding'] = df['descripcion'].map(EMB_CACHE)

    # Filtrar filas sin embedding (por seguridad)
    df = df[~df['embedding'].isnull()]
    df_utpl = df[df['nombre_ies'] == 'universidad tecnica particular de loja']
    df_ext = df[df['nombre_ies'] != 'universidad tecnica particular de loja']

    combinaciones = df_utpl[['campo_amplio', 'nivel_formacion']].drop_duplicates()
    resultados = []

    for _, fila in combinaciones.iterrows():
        campo, nivel = fila['campo_amplio'], fila['nivel_formacion']
        bloque_utpl = df_utpl[(df_utpl['campo_amplio'] == campo) & (df_utpl['nivel_formacion'] == nivel)]
        bloque_ext = df_ext[(df_ext['campo_amplio'] == campo) & (df_ext['nivel_formacion'] == nivel)]

        if bloque_utpl.empty or bloque_ext.empty:
            continue

        try:
            emb_utpl = torch.stack(bloque_utpl['embedding'].tolist())
            emb_ext = torch.stack(bloque_ext['embedding'].tolist())
        except Exception as e:
            print(f"❌ Error al hacer stack en bloque {campo}-{nivel}: {e}")
            continue

        sim_matrix = util.cos_sim(emb_utpl, emb_ext).cpu().numpy()

        for i, u_row in enumerate(bloque_utpl.itertuples(index=False)):
            for j, e_row in enumerate(bloque_ext.itertuples(index=False)):
                score = sim_matrix[i, j]
                if score < SIM_THRESHOLD:
                    continue
                resultados.append({
                    "id_emparejamiento": f"{u_row.id}_{e_row.id}",

                    # UTPL
                    "id_utpl": u_row.id,
                    "nombre_carrera_utpl": u_row.nombre_carrera,
                    "descripcion_utpl": u_row.descripcion,
                    "campo_amplio_utpl": u_row.campo_amplio,
                    "nivel_formacion_utpl": u_row.nivel_formacion,
                    "nombre_ies_utpl": "universidad tecnica particular de loja",
                    "modalidad_utpl": u_row.modalidad,
                    "provincia_utpl": u_row.provincia,

                    # Externa
                    "id_externa": e_row.id,
                    "nombre_carrera_externa": e_row.nombre_carrera,
                    "descripcion_externa": e_row.descripcion,
                    "campo_amplio_externa": e_row.campo_amplio,
                    "nivel_formacion_externa": e_row.nivel_formacion,
                    "nombre_ies_externa": e_row.nombre_ies,
                    "modalidad_externa": e_row.modalidad,
                    "provincia_externa": e_row.provincia,

                    # Métrica
                    "similitud_embeddings": round(score * 100, 2)
                })

    return pd.DataFrame(resultados)
