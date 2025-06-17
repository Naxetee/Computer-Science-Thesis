import json
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sensitive_attribute = 'income'

# Configura los modelos y sus grids
MODELS = {
    "LogisticRegression": (LogisticRegression(max_iter=1000), {
        "C": [0.1, 10, 100],
        "penalty": ["l1", "l2", None],
        "solver": ["lbfgs", "liblinear"]
    }),
    "KNeighbors": (KNeighborsClassifier(), {
        "n_neighbors": [3, 7, 11],
        "weights": ["uniform", "distance"]
    }),
    "XGBClassifier": (XGBClassifier(eval_metric='logloss'), {
        "n_estimators": [50, 100],
        "max_depth": [3, 7],
        "learning_rate": [0.01, 0.1]
    }),
    "RandomForest": (RandomForestClassifier(), {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7, None],
        "criterion": ["gini", "entropy", "log_loss"]
    })
}

def get_best_model(input_path: str, output_path: str = '', save_results: bool = False, debug_comments: bool = False) -> dict:
    """
    Obtiene el mejor modelo entrenado con los datos originales usando GridSearchCV.

    Args:
        input_path (str): Ruta del dataset CSV preprocesado.
        output_path (str, optional): Ruta para guardar los resultados.
        save_results (bool, optional): Si es True, guarda los resultados en un archivo CSV.
        debug_comments (bool, optional): Si es True, imprime información de depuración.

    Returns:
        dict: Diccionario con información del mejor modelo encontrado.
    """

    
    # Verifica si el archivo de entrada existe, si no, devuelve un error
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"El archivo de entrada {input_path} no existe. Por favor, crea el directorio antes de ejecutar el script.")
    
    # Verifica si el directorio de salida existe, si no, lo crea
    if output_path and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if debug_comments: print(f"Directorio de salida creado: {os.path.dirname(output_path)}")

    # Cargar el dataset
    try:
        df = pd.read_csv(input_path)
        if debug_comments: print(f"Dataset cargado: {input_path.split('/')[-1]}")
    except Exception as e:
        raise ValueError(f"Error al cargar el dataset: {e}")

    # Separar características y etiqueta
    X = df.drop(columns=[sensitive_attribute])
    y = df[sensitive_attribute]
    
    # Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Almacenar los resultados de cada modelo
    results = {}

    if debug_comments:
        print("\n\n|_______________________________|")
        print("|_____Entrenamiento Inciado_____|")
        print("|_______________________________|\n\n")

    # Iterar sobre los modelos
    for model_name, (model, param_grid) in MODELS.items():
        if debug_comments: print(f"Entrenando {model_name}...")
        try:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Evaluar el modelo
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            # Calcular métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
        except Exception as e:
            if debug_comments: print(f"Error al entrenar el modelo {model_name}: {e}")
            continue

        if debug_comments:
            print(f"Model: {model_name} | Best Params: {grid_search.best_params_} | Accuracy: {accuracy:.4f}")

        # Almacenar resultados
        results[model_name] = {
            "best_params": grid_search.best_params_,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        }

    # Identificar el mejor modelo basado en la precisión
    best_model_name, best_model_info = max(results.items(), key=lambda x: x[1]['metrics']['accuracy'])
    result_dict = {
        "dataset": os.path.basename(input_path),
        "best_model": best_model_name,
        "best_params": best_model_info['best_params'],
        "metrics": best_model_info['metrics']
    }

    if debug_comments:
        print(f"\tMejor modelo: {best_model_name}")
        print(f"\tParámetros: {result_dict['best_params']}")
        print(f"\tMétricas: {result_dict['metrics']}")

    # Guardar resultados en CSV si se especifica
    if save_results:
        try:
            pd.DataFrame([{
                "dataset": result_dict["dataset"],
                "best_model": result_dict["best_model"],
                "best_params": json.dumps(result_dict["best_params"]),
                "metrics": json.dumps(result_dict["metrics"])
            }]).to_csv(output_path, index=False)
            if debug_comments: print(f"Resultados guardados en: {output_path}")
        except Exception as e:
            raise ValueError(f"Error al guardar los resultados: {e}")

    if debug_comments:
        print("\n\n|__________________________________|")
        print("|_____Entrenamiento Finalizado______|")
        print("|___________________________________|\n\n")
    return result_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo original con GridSearchCV")
    parser.add_argument('--input_path', type=str, default='./data/adults/preprocessed/original/adults_original.csv', help='Ruta del dataset CSV preprocesado')
    parser.add_argument('--output_path', type=str, default='./data/adults/results/models/best_model_over_original.csv', help='Ruta de salida para guardar el mejor modelo')
    parser.add_argument('--save_results', action='store_true', help='Guardar resultados en un archivo CSV')
    parser.add_argument('--debug_comments', action='store_true', help='Imprimir comentarios de depuración')

    args = parser.parse_args()

    if args.debug_comments:
        print("\n🔍 Modo depuración activado")
        input("Presiona Enter para continuar...")

    # Llamar a la función principal
    best_model = get_best_model(args.input_path, args.output_path, args.save_results, args.debug_comments)
