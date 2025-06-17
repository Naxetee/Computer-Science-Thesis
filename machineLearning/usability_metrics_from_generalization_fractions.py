import os
import re # Expresiones regulares
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import argparse

# Asociación de posibles modelos resultantes del GridSearch
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNeighbors": KNeighborsClassifier(),
    "XGBClassifier": XGBClassifier(eval_metric='logloss'),
    "RandomForest": RandomForestClassifier()
}

def get_hierarchy_dict(levels: str) -> Dict[str, int]:
    """
    Obtiene un diccionario con la jerarquía de niveles proveniente de una cadena de la forma:
    \n**Ejemplo**: *"{'level1': '1/3', 'level2': '2/3', 'level3': '0/3'}"*.
    \nDevuelve un diccionario con los niveles y sus valores, filtrando aquellos con valor 0.
    Args:
        levels (str): Cadena que representa los niveles y sus valores en formato 'nivel: valor/total'.
    Returns:
        dict: Diccionario con los niveles y sus valores, excluyendo aquellos con valor 0.
    """
    levels = re.findall(r"'[\w-]+': '\d/\d'", levels)  
    hier_dict =  {k.strip("'"): int(l.strip("'").split("/")[0]) for k, l in (x.split(': ') for x in levels)}
    return {k: v for k, v in hier_dict.items() if v > 0}

def load_hierarchies(hier_path:str) -> Dict[str, Dict[str, list]]:
    """
    Carga las jerarquías de atributos desde archivos CSV en el directorio especificado.
    Cada archivo debe tener un nombre del formato 'atributo_jerarquia.csv'.
    Devuelve un diccionario donde las claves son los nombres de los atributos y los valores son DataFrames.
    Args:
        hier_path (str): Ruta al directorio que contiene los archivos de jerarquías.
    Returns:
        dict: Diccionario de pares atributo:diccionario de jerarquía. Cada diccionario de jerarquía
        tiene, para cada valor original, una lista de valores según su nivel de generalización.
    Raises:
        FileNotFoundError: Si el directorio especificado no existe.
    """
    if not os.path.exists(hier_path):
        raise FileNotFoundError(f"El directorio {hier_path} no existe.")
    # Cargamos para cada atributo su jerarquía
    hierarchies = {}
    for f in os.listdir(hier_path):
        if f.endswith('.csv'):
            attr = f.split('_')[0]  # Asumimos que el nombre del archivo es 'atributo_jerarquia.csv'
            hier_df = pd.read_csv(os.path.join(hier_path, f), index_col=0)
            # Convertimos el DataFrame a un diccionario donde el índice del DataFrame es la clave y la fila es el valor
            hierarchies[attr] = hier_df.to_dict(orient='dict')
    return hierarchies

def rebuild_dataset(original_df: pd.DataFrame, hierarchies: Dict[str, pd.DataFrame], levels: Dict[str, int]) -> pd.DataFrame:
    """
    Reconstruye un DataFrame original utilizando las jerarquías de atributos proporcionadas.
    Args:
        original_df (pd.DataFrame): DataFrame original que contiene los atributos a reconstruir.
        hierarchies (Dict[str, pd.DataFrame]): Diccionario con las jerarquías de atributos.
        levels (Dict[str, int]): Diccionario con los niveles y sus valores.
    Returns:
        pd.DataFrame: DataFrame reconstruido con los atributos modificados según las jerarquías.
        
    Raises:
        ValueError: Si alguna jerarquía de atributo no se encuentra en las jerarquías proporcionadas.
    """
    
    # Tomamos los atributos presentes en levels
    modified_attr = [attrs for attrs in original_df.columns if attrs in levels.keys()]
    rest_attrs = [attrs for attrs in original_df.columns if attrs not in modified_attr]

    # Creamos un DataFrame vacío para almacenar los datos reconstruidos
    rebuilt_df = pd.DataFrame()

    # Añadimos las columnas de los atributos no modificados directamente sin modificación
    for attr in rest_attrs:
        rebuilt_df[attr] = original_df[attr]

    # Recorremos los atributos modificados y aplicamos las jerarquías
    for attr in modified_attr:
        if attr in hierarchies:
            # Obtenemos la jerarquía y el nivel de generalización del atributo
            hierarchy = hierarchies[attr]
            l = str(levels[attr])

            # Mapeamos los valores del atributo a sus jerarquías
            rebuilt_df[attr] = original_df[attr].map(lambda x: hierarchy[l][x])
        else:
            raise ValueError(f"La jerarquía para el atributo '{attr}' no se encuentra en las jerarquías proporcionadas.")
    
    # Aseguramos que el orden de las columnas sea el mismo que el original
    return rebuilt_df[original_df.columns]  # Aseguramos que el orden de las columnas sea el mismo que el original

def preprocess_dataset(df: pd.DataFrame, target_col:str) -> pd.DataFrame:
    """
    Preprocesa un DataFrame para dejarlo listo para la actuación de la IA
    Args:
        df (pd.DataFrame): DataFrame a preprocesar.
        target_col (str): Nombre de la columna objetivo que se quiere predecir.
    Returns:
        pd.DataFrame: DataFrame preprocesado.
    """
    # Separa la columna objetivo del resto de los datos y la convertimos a binario
    y = df[target_col].apply(lambda x: 1.0 if x == ">50K" else 0.0)
    X = df.drop(target_col, axis=1)

    # Detecta columnas categóricas automáticamente
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Aplica Label Encoding a las columnas categóricas
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])

    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
        ('num', StandardScaler(), numeric_cols + categorical_cols)  # Incluye las columnas categóricas ya codificadas
        ],
        remainder='passthrough'  # Para mantener las columnas no numéricas
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    X_processed = pipeline.fit_transform(X)

    df_processed = pd.DataFrame(X_processed, columns=X.columns)
    df_processed[target_col] = y.values

    return df_processed

def evaluate_model(model_path:str, df:pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, float]:
    """
    Entrena un modelo guardado en la ruta especificada utilizando el DataFrame proporcionado y lo valida con el DataFrame original.
    Args:
        model_path (str): Ruta al archivo del modelo guardado.
        df (pd.DataFrame): DataFrame con los datos de prueba.
        original_df (pd.DataFrame): DataFrame original para validar el modelo.
    Returns:
        dict: Diccionario con las métricas de evaluación del modelo.
    """
    # Cargamos el mejor modelo y los mejores parámetros desde el archivo CSV
    best_model_df = pd.read_csv(model_path)
    best_model = best_model_df["best_model"].values[0]
    best_params = eval(best_model_df["best_params"].values[0])

    # Creamos una instancia del modelo con los mejores parámetros
    model = MODELS[best_model]
    model.set_params(**best_params)

    # Separamos las características y la etiqueta
    X = df.drop(columns=["income"])
    y = df["income"]
    X_original = original_df.drop(columns=["income"])
    y_original = original_df["income"]

    # Aseguramos de que las columnas están en el mismo orden
    X = X[X_original.columns]

    # Entrena y predice
    model.fit(X, y)
    y_pred = model.predict(X_original)

    # Calculamos las métricas
    metrics = {
        "accuracy": accuracy_score(y_original, y_pred),
        "precision": precision_score(y_original, y_pred),
        "recall": recall_score(y_original, y_pred),
        "f1_score": f1_score(y_original, y_pred)
    }
    return metrics
    

def iterate_files(anon_results_path: str, usability_results_path: str, hier_path:  str,
                original_dataset_path: str, best_model_path: str, debug_comments: bool = False) -> None:
    """
    Itera sobre los archivos de resultados de anonimización, reconstruye los datasets y evalúa su usabilidad.
    Args:
        anon_results_path (str): Ruta a los resultados de anonimización.
        usability_results_path (str): Ruta a los resultados de usabilidad.
        hier_path (str): Ruta a las jerarquías de atributos.
        original_dataset_path (str): Ruta al dataset original.
        best_model_path (str): Ruta al mejor modelo guardado.
    Returns:
        None
    Raises:
        FileNotFoundError: Si el archivo de usabilidad no existe y no se puede crear.
        ValueError: Si alguna jerarquía de atributo no se encuentra en las jerarquías proporcionadas.
    """
    # Verificamos que las rutas existen
    if not os.path.exists(usability_results_path):
        os.makedirs(usability_results_path)
        if debug_comments: print(f"📂 Creada carpeta de resultados de usabilidad en: {usability_results_path}")

    # Cargamos el dataset original y lo preprocesamos
    original_df = pd.read_csv(original_dataset_path)
    processed_original_df = preprocess_dataset(original_df, 'income')

    # Cargamos las jerarquías
    hierarchies = load_hierarchies(hier_path)

    # Tomamos los archivos de la forma k_{k}-l_{l}-t_{t}-results.csv
    pattern = re.compile(r'k_\d+-l_\d+-t_(\d+\.\d+|\d+)-results\.csv')

    files = [f for f in os.listdir(anon_results_path) if pattern.fullmatch(f)]

    for f in files:
        if debug_comments: print(f"🔄 Procesando archivo: {f}")
        usability_file_path = os.path.join(usability_results_path, f)

        # Verificamos si el archivo de usabilidad existe
        if not os.path.exists(usability_file_path):
            # Si no existe, inicializamos una lista para almacenar los QIDs procesados y un DataFrame vacío para las métricas
            processed_qids = []
            metrics_df = pd.DataFrame(columns=['QIDs', 'accuracy', 'precision', 'recall', 'f1_score'])
        else:
            # Si el archivo ya existe, lo leemos para evitar duplicados
            metrics_df = pd.read_csv(usability_file_path)
            processed_qids = [set(d.split(", ")) for d in metrics_df["QIDs"].unique().tolist()]

        # Leemos el archivo
        current_df = pd.read_csv(os.path.join(anon_results_path, f))

        # Reconstruimos cada dataset
        for r in current_df.itertuples():
            qids, levels = r.QIDs, get_hierarchy_dict(r._3)
            if set(qids.split(", ")) in processed_qids:
                # Si ya hemos procesado este QID, lo saltamos
                if debug_comments: print(f"⏭️ QIDs {qids} ya procesados, saltando...")
                continue

            # Reconstruimos el dataset
            rebuilt_df = rebuild_dataset(original_df, hierarchies, levels)

            # Ahora lo preprocesamos para calcular las métricas de usabilidad
            processed_df = preprocess_dataset(rebuilt_df, 'income')

            # Evaluamos el modelo
            metrics = evaluate_model(best_model_path, processed_df, processed_original_df)
            
            # Guardamos los resultados de usabilidad
            new_metrics = pd.DataFrame({
                'QIDs': qids,
                'accuracy': [metrics['accuracy']],
                'precision': [metrics['precision']],
                'recall': [metrics['recall']],
                'f1_score': [metrics['f1_score']]
            })
            metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)
            
        # Guardamos el DataFrame de métricas actualizado
        metrics_df.to_csv(usability_file_path, index=False)
        if debug_comments: print(f"✅ Resultados de usabilidad guardados en: 📄{usability_file_path}\n\n")
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparar usabilidad vs anonimización en datasets.")
    parser.add_argument('--anon_results_path', type=str, default='./data/adults/results/anonymization/', help='Ruta a los resultados de anonimización')
    parser.add_argument('--usability_results_path', type=str, default='./data/adults/results/usability/', help='Ruta a los resultados de usabilidad')
    parser.add_argument('--hier_path', type=str, default='./data/adults/hierarchies/', help='Ruta a las jerarquías')
    parser.add_argument('--original_dataset_path', type=str, default='./data/adults/original/adults_rounded.csv', help='Ruta al dataset original')
    parser.add_argument('--best_model_path', type=str, default='./data/adults/results/models/best_model_over_original.csv', help='Ruta al mejor modelo')
    parser.add_argument('--q', type=int, default=13, help='Número de QIDs a considerar para la anonimización')
    parser.add_argument('--debug_comments', action='store_true', help='Activar comentarios de depuración')
    args = parser.parse_args()

    if args.debug_comments:
        print("🔧 Modo depuración activado. Se mostrarán comentarios detallados durante la ejecución.")
        input("Presiona cualquier tecla para continuar...")

    iterate_files(
        anon_results_path = f"{args.anon_results_path}{args.q}_qids/",
        usability_results_path = f"{args.usability_results_path}{args.q}_qids/",
        hier_path = args.hier_path,
        original_dataset_path = args.original_dataset_path,
        best_model_path = args.best_model_path,
        debug_comments = args.debug_comments
    )
