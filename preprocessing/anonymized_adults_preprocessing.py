import pandas as pd
import numpy as np
import os
import argparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

def preprocess_adult_datasets(input_folder : str, output_folder : str, debug_comments: bool = False) -> None:
    """
    Preprocesa los datasets de adultos anonimizados, aplicando OneHotEncoding a las columnas categóricas.

    Args:
        - input_folder: str
            Carpeta de entrada donde se encuentran los datasets CSV. Si isOriginal es True, se asume que input_folder contiene un solo dataset.
        - output_folder: str
            Carpeta de salida donde se guardarán los datasets preprocesados.
        - debug_comments: bool, opcional
            Si es True, imprime información de depuración.

    Returns:
        None
    """

    if debug_comments: 
        print("\n\n|_______________________________|")
        print("|___Preprocesamiento Inciado____|")
        print("|_______________________________|\n\n")

    if not os.path.exists(output_folder):
        # Crea la carpeta de salida si no existe
        os.makedirs(output_folder, exist_ok=True)
        processed_datasets = set([])
    else:
        # Guardamos todas las combinaciones ya procesadas
        processed_datasets = [set(f.replace(".csv", "").split("_")) for f in os.listdir(output_folder) if f.endswith('.csv')]

    # Detecta todos los CSVs de entrada
    dataset_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]

    for path in dataset_paths:
        qids = set(os.path.basename(path).replace(".csv", "").split("_"))
        # Verifica si el dataset ya ha sido procesado
        if qids in processed_datasets:
            if debug_comments: print(f"🔍 '{os.path.basename(path)}'ya ha sido procesado, pasando al siguiente...")
            continue
        if debug_comments: print(f"🔍 Procesando '{path}'")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            if debug_comments: print(f"❌ Error al leer '{path}': {e}")
            continue

        target_col = "income"
        if target_col not in df.columns:
            if debug_comments: print(f"❌ '{path}' no tiene columna objetivo '{target_col}'. Saltando.")
            continue

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

        # Guarda el nuevo dataset
        try:
            output_path = os.path.join(output_folder, os.path.basename(path))
            processed_datasets.append(qids)
            df_processed.to_csv(output_path, index=False)
            if debug_comments: print(f"✅ Guardado: {output_path}")
        except Exception as e:
            if debug_comments: print(f"❌ Error exportando el dataset: {output_path} : {e}")
            continue
    if debug_comments: 
        print("\n\n|_______________________________|")
        print("|__Preprocesamiento completado__|")
        print("|_______________________________|\n\n")

    
# Ejecutar la función principal que tome argumentos de la línea de comandos
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocesar datasets de adultos anonimizados")
    parser.add_argument('--debug_comments', action='store_true', help="Activar modo depuración")
    parser.add_argument('--isOriginal', action='store_true', help="Indica si el dataset es original y no anonimo")

    args = parser.parse_args()

    if args.debug_comments:
        print("\n🔍 Modo depuración activado")
        if args.isOriginal:
            print("Procesando dataset original")
        else:
            print("Procesando datasets anónimos")
        input("Presiona Enter para continuar...")

    if args.isOriginal:
        input_path = "./data/adults/original/"
        output_path = "./data/adults/preprocessed/original/"
    else:
        input_path = "./data/adults/anonymized/"
        output_path = "./data/adults/preprocessed/"

    preprocess_adult_datasets(
        input_folder=input_path,
        output_folder=output_path,
        debug_comments=args.debug_comments
    )