import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Configura los modelos y sus grids
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNeighbors": KNeighborsClassifier(),
    "XGBClassifier": XGBClassifier(eval_metric='logloss'),
    "RandomForest": RandomForestClassifier()
}

def evaluate_model(model_path: str, datasets_path: str, original_dataset_path: str, output_path : str, debug_comments: bool = False) -> None:
    """
    Evalúa el modelo que mejor actúa en el dataset original, en múltiples datasets y guarda las métricas en un archivo CSV.

    Args:
        model_path (str): Ruta a la información del mejor modelo.
        datasets_path (str): Ruta a la carpeta que contiene los datasets.
        original_dataset_path (str): Ruta al dataset original.
        output_path (str): Ruta del csv donde se guardarán las métricas.
        debug_comments (bool): Si es True, imprime información de depuración.

    Returns:
        None
    """

    # Carga el modelo
    try:
        best_model_df = pd.read_csv(model_path)
        best_model = best_model_df["best_model"].values[0]
        best_params = eval(best_model_df["best_params"].values[0])
        if debug_comments: 
            print(f"Mejor modelo cargado: {best_model} | Parámetros: {best_params}")
    except Exception as e:
        raise ValueError(f"Error al cargar el mejor modelo: {e}")
    
    # Función para reiniciar el modelo y establecer los parámetros correspondientes
    def model_reset(model_name=best_model, params=best_params):
        model = MODELS[model_name]
        model.set_params(**params)
        return model
    
    # Comprueba la existencia del archivo de salida y lo carga o lo crea si no existe
    try:
        if os.path.exists(output_path):
            metrics_df = pd.read_csv(output_path)
            processed_datasets = [set(d.split(", ")) for d in metrics_df["dataset"].unique().tolist()]
        else:
            metrics_df = pd.DataFrame(columns=["dataset", "accuracy", "precision", "recall", "f1_score", "path"])
            processed_datasets = []
    except Exception as e:
        raise ValueError(f"Error al manejar el archivo de salida: {e}")
    
    # Obtiene el dataset original
    try:
        original_df = pd.read_csv(original_dataset_path)
        X_original = original_df.drop(columns=["income"])
        y_original = original_df["income"]
    except Exception as e:
        raise ValueError(f"Error al cargar el dataset original: {e}")

    # Itera sobre todos los datasets en la carpeta
    for dataset_file in [f for f in os.listdir(datasets_path) if f.endswith(".csv")]:
            # Verifica si el dataset ya ha sido procesado
            qids = set(dataset_file.replace(".csv", "").split("_"))

            # Si el dataset ya ha sido procesado, lo salta
            if qids in processed_datasets:
                if debug_comments: print(f" Dataset {qids} ya procesado, saltando...")
                continue
            
            if debug_comments: print(f"\nEvaluando el dataset: {dataset_file}")

            # Ruta completa del dataset
            dataset_path = os.path.join(datasets_path, dataset_file)

            try:
                # Carga el dataset anonimizado
                df = pd.read_csv(dataset_path)

                # Separa las características y la etiqueta
                X = df.drop(columns=["income"])
                y = df["income"]

                # Añade las columnas faltantes con valores 0
                for col in X_original.columns:
                    if col not in X.columns:
                        X[col] = 0

                # Asegura de que las columnas están en el mismo orden
                X = X[X_original.columns]

                # Reinicia el modelo
                model = model_reset()

                # Entrena y predice
                model.fit(X, y)
                y_pred = model.predict(X_original)

            except Exception as e:
                if debug_comments: print(f"Error al procesar el dataset {dataset_file}: {e}")
                continue

            # Calcula las métricas
            try:
                accuracy = accuracy_score(y_original, y_pred)
                precision = precision_score(y_original, y_pred)
                recall = recall_score(y_original, y_pred)
                f1 = f1_score(y_original, y_pred)
            except Exception as e:
                if debug_comments: print(f"Error al calcular métricas para {dataset_file}: {e}")
                continue

            # Guarda las métricas en el DataFrame
            new_metrics = pd.DataFrame([{
                "dataset": ", ".join(qids),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "path": dataset_path
            }])
            metrics_df = pd.concat([metrics_df, new_metrics], ignore_index=True)

            if debug_comments: print(f" Dataset {dataset_file} procesado.")

            # Guarda las métricas en un archivo CSV
            metrics_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa un modelo en múltiples conjuntos de datos.")
    parser.add_argument("--model_path", type=str, required=False, default="./data/adults/results/models/best_model_original.csv", help="Ruta al archivo del modelo entrenado (por defecto: mejor_modelo.csv).")
    parser.add_argument("--datasets_path", type=str, required=False, default="./data/adults/preprocessed/", help="Ruta al directorio que contiene los conjuntos de datos (por defecto: datasets/).")
    parser.add_argument("--original_dataset_path", type=str, required=False, default="./data/adults/preprocessed/original/adults_original.csv", help="Ruta al dataset original (por defecto: dataset_original.csv).")
    parser.add_argument("--output_path", type=str, required=False, default="./data/adults/results/usability/usability_anonymized.csv", help="Ruta para guardar el archivo CSV con las métricas de evaluación (por defecto: metricas.csv).")
    parser.add_argument("--debug_comments", action="store_true", help="Habilitar comentarios de depuración.")
    args = parser.parse_args()

    if args.debug_comments:
        print("\n🔍 Modo depuración activado")
        input("Presiona Enter para continuar...")

    evaluate_model(
        model_path=args.model_path,
        datasets_path=args.datasets_path,
        original_dataset_path=args.original_dataset_path,
        output_path=args.output_path,
        debug_comments=args.debug_comments
    )