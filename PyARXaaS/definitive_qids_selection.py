from pyarxaas import ARXaaS
from pyarxaas.privacy_models import KAnonymity, LDiversityDistinct, TClosenessEqualDistance
import pandas as pd
from utils import iterateCombinations
import argparse
import os

def qids_selection( 
                    sensitive_attribute: str,
                    mandatory_qids: list,
                    insensitive_attributes: list,
                    max_num_of_qids: int,
                    dataset_path: str, 
                    output_folder: str,
                    hierarchy_folder: str,
                    output_file: str,
                    ARXaaS_port: int,
                    debug : bool, 
                    export : bool
                    ) -> None:
    """
    Función principal para la selección de QIDs y ejecución de ARXaaS.

    :param sensitive_attribute: Atributo objetivo a predecir.
    :param mandatory_qids: Lista de atributos quasi-identificadores obligatorios.
    :param insensitive_attributes: Lista de atributos que no son potenciales QIDs ni sensibles.
    :param max_num_of_qids: Número de atributos que serán quasiidentificadores incluyendo los obligatorios
    :param dataset_path: Ruta al dataset a procesar.
    :param output_folder: Carpeta de salida donde se exportarán los datasets anonimizados.
    :param output_file: Nombre del archivo CSV donde se guardarán los resultados.
    :param ARXaaS_port: Puerto de la instancia de ARXaaS corriendo en local. Por defecto es 8080.
    :param debug: Si es True, imprime información de depuración.
    :param export: Si es True, exporta los datasets anonimizados a CSV.
    """
    # Cargar el dataset
    dataset_name = os.path.basename(dataset_path)
    dataset_path = os.path.dirname(dataset_path) + '/'

    if not os.path.exists(dataset_path + dataset_name):
        raise ValueError(f"❌ El archivo '{dataset_name}' no se encuentra en la ruta '{dataset_path}'.")
    else:
        df = pd.read_csv(dataset_path + dataset_name)
        if debug: print(f"📋 Dataset '{dataset_name}' cargado")

    # Comprobamos que todos los atributos especificados existen en el dataset
    if any(atr not in df.columns for atr in [sensitive_attribute] + mandatory_qids + insensitive_attributes):
        raise ValueError("❌ Algunos de los atributos especificados no existen en el dataset.")

    # Definir los atributos candidatos a quasi-identificadores sin contar los obligatorios
    potential_qids = [atr for atr in df.columns if atr not in [sensitive_attribute] + insensitive_attributes]

    # Definimos los modelos de privacidad a utilizar
    privacy_models = [
        KAnonymity(5),
        LDiversityDistinct(2, sensitive_attribute),
        TClosenessEqualDistance(0.5, sensitive_attribute)
    ]

    # Conectar con ARXaaS
    try:
        arxaas = ARXaaS(f"http://localhost:{ARXaaS_port}")
        if debug: print("✅ Conexión con ARXaaS establecida")
    except Exception as e:
        raise RuntimeError("❌ Error al conectar con ARXaaS:", e)

    iterateCombinations(
        arxaas=arxaas,
        df=df,
        mandatory_qids=mandatory_qids,
        potential_qids=potential_qids,
        sensitive_attribute=sensitive_attribute,
        insensitive_attributes=insensitive_attributes,
        privacy_models=privacy_models,
        hierarchy_folder = hierarchy_folder,
        output_file=output_file,
        max_num_of_qids=max_num_of_qids,
        debug_comments=debug,
        export_datasets=export,
        output_folder=output_folder
    )

    if debug: print("✅ Proceso de selección de QIDs finalizado")


# Ejecutar la función principal que tome argumentos de la línea de comandos
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seleccionar QIDs y ejecutar ARXaaS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--sensitive_attribute', type=str, default='income', help="Atributo objetivo a predecir")
    parser.add_argument('--mandatory_qids', nargs='+', default=['sex', 'age'], help="Lista de atributos quasi-identificadores obligatorios")
    parser.add_argument('--insensitive_attributes', nargs='+', default=['fnlwgt'], help="Lista de atributos que no son potenciales QIDs ni sensibles")
    parser.add_argument('--max_num_of_qids', type=int, default=4, help="Número máximo de atributos que serán elegidos QIDs contando a los obligatorios.")
    parser.add_argument('--dataset_path', type=str, default='./data/adults/original/adults_rounded.csv', help="Ruta al dataset a procesar")
    parser.add_argument('--output_folder', type=str, default='./data/adults/anonymized/', help="Carpeta de salida donde se exportarán los datasets anonimizados")
    parser.add_argument('--hierarchy_folder', type=str, default='./data/adults/hierarchies/', help="Carpeta donde se encuentran las jerarquías")
    parser.add_argument('--output_file', type=str, default="./data/adults/results/anonymization/anonymization_results.csv", help="Nombre del archivo CSV donde se guardarán los resultados")
    parser.add_argument('--ARXaaS_port', type=int, default=8080, help="Puerto de la instancia de ARXaaS corriendo en local")
    parser.add_argument('--debug', action='store_true', help="Activar modo depuración")
    parser.add_argument('--export', action='store_true', help="Exportar datasets anonimizados a csv")

    args = parser.parse_args()

    if args.debug:
        print("🔍 Modo depuración activado")

    # Llamar a la función principal con los argumentos proporcionados
    qids_selection(
        sensitive_attribute=args.sensitive_attribute,
        mandatory_qids= [], # args.mandatory_qids,
        insensitive_attributes = [], # args.insensitive_attributes,
        dataset_path=args.dataset_path,
        output_folder=args.output_folder,
        max_num_of_qids=args.max_num_of_qids,
        hierarchy_folder=args.hierarchy_folder,
        output_file=args.output_file,
        ARXaaS_port=args.ARXaaS_port,
        debug=args.debug,
        export=args.export
    )