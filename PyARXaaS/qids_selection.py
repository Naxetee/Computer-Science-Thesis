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
                    anonymity_models: dict,
                    dataset_path: str, 
                    output_folder: str,
                    hierarchy_folder: str,
                    output_file: str,
                    ARXaaS_port: int,
                    debug : bool, 
                    export : bool
                    ) -> None:
    """
    Funcion principal para la seleccion de QIDs y ejecucion de ARXaaS.
    Args:
        sensitive_attribute (str) : Atributo objetivo a predecir.
        mandatory_qids (list) : Lista de atributos quasi-identificadores obligatorios.
        insensitive_attributes (list) : Lista de atributos que no son potenciales QIDs ni sensibles.
        max_num_of_qids (int) : Numero de atributos que seran quasiidentificadores incluyendo los obligatorios.
        anonymity_models (dict) : Diccionario con los modelos de privacidad a utilizar.
        dataset_path (str) : Ruta al dataset a procesar.
        output_folder (str) : Carpeta de salida donde se exportaran los datasets anonimizados.
        output_file (str) : Nombre del archivo CSV donde se guardaran los resultados.
        hierarchy_folder (str) : Carpeta donde se encuentran las jerarquias.
        ARXaaS_port (int) : Puerto de la instancia de ARXaaS corriendo en local. Por defecto es 8080.
        debug (bool) : Si es True, imprime informacion de depuracion.
        export (bool) : Si es True, exporta los datasets anonimizados a CSV.
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
    privacy_models = []
    if anonymity_models.get('k',0):
        k = anonymity_models['k']
        privacy_models.append(KAnonymity(k))
    if anonymity_models.get('l',0):
        l = anonymity_models['l']
        privacy_models.append(LDiversityDistinct(l, sensitive_attribute))
    if anonymity_models.get('t',0):
        t = anonymity_models['t']
        privacy_models.append(TClosenessEqualDistance(t, sensitive_attribute))

    # Conectar con ARXaaS
    try:
        arxaas = ARXaaS(f"http://localhost:{ARXaaS_port}")
        if debug: print("✅ Conexion con ARXaaS establecida")
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

    if debug: print("✅ Proceso de seleccion de QIDs finalizado")


# Ejecutar la funcion principal que tome argumentos de la linea de comandos
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seleccionar QIDs y ejecutar ARXaaS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--sensitive_attribute', type=str, default='income', help="Atributo objetivo a predecir")
    parser.add_argument('--mandatory_qids', nargs='+', default=['sex', 'age'], help="Lista de atributos quasi-identificadores obligatorios")
    parser.add_argument('--insensitive_attributes', nargs='+', default=['fnlwgt'], help="Lista de atributos que no son potenciales QIDs ni sensibles")
    parser.add_argument('--max_num_of_qids', type=int, default=4, help="Numero maximo de atributos que seran elegidos QIDs contando a los obligatorios.")
    parser.add_argument('--k_anonymity', type=int, default=5, help="Valor de k para el modelo de K-Anonimidad")
    parser.add_argument('--l_diversity', type=int, default=2, help="Valor de l para el modelo de L-Diversidad")
    parser.add_argument('--t_closeness', type=int, default=0.5, help="Valor de t para el modelo de T-Cercania")
    parser.add_argument('--dataset_path', type=str, default='./data/adults/original/adults_rounded.csv', help="Ruta al dataset a procesar")
    parser.add_argument('--output_folder', type=str, default='./data/adults/anonymized/', help="Carpeta de salida donde se exportaran los datasets anonimizados")
    parser.add_argument('--hierarchy_folder', type=str, default='./data/adults/hierarchies/', help="Carpeta donde se encuentran las jerarquias")
    parser.add_argument('--output_file', type=str, default="./data/adults/results/anonymization/anonymization_results.csv", help="Nombre del archivo CSV donde se guardaran los resultados")
    parser.add_argument('--ARXaaS_port', type=int, default=8080, help="Puerto de la instancia de ARXaaS corriendo en local")
    parser.add_argument('--debug', action='store_true', help="Activar modo depuracion")
    parser.add_argument('--export', action='store_true', help="Exportar datasets anonimizados a csv")

    args = parser.parse_args()

    if args.debug:
        print("🔍 Modo depuracion activado")

    # Llamar a la funcion principal con los argumentos proporcionados
    qids_selection(
        sensitive_attribute=args.sensitive_attribute,
        mandatory_qids= args.mandatory_qids,
        insensitive_attributes = args.insensitive_attributes,
        anonymity_models= {
            'k': args.k_anonymity,
            'l': args.l_diversity,
            't': args.t_closeness
        },
        dataset_path=args.dataset_path,
        output_folder=args.output_folder,
        max_num_of_qids=args.max_num_of_qids,
        hierarchy_folder=args.hierarchy_folder,
        output_file=args.output_file,
        ARXaaS_port=args.ARXaaS_port,
        debug=args.debug,
        export=args.export
    )