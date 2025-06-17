import os
import pandas as pd
from pyarxaas import ARXaaS, AttributeType, Dataset
from itertools import combinations
from typing import List, Dict, Tuple, Set

def iterateCombinations(
                        arxaas : ARXaaS , 
                        df : pd.DataFrame, 
                        mandatory_qids : List[str], 
                        potential_qids : List[str], 
                        sensitive_attribute : str,
                        insensitive_attributes : List[str], 
                        privacy_models : List[object], 
                        output_file : str,
                        hierarchy_folder: str,
                        max_num_of_qids : int, 
                        output_folder: str,
                        export_datasets : bool,
                        debug_comments : bool, 
                        ) -> None:
    """
    Función para iterar sobre todas las combinaciones de QIDs y realizar la anonimización. Los resultados se guardan en el archivo :param output_file.

    :param arxaas: Conexión a ARXaaS
    :param df: DataFrame con los datos originales
    :param potential_qids: Lista de atributos candidatos a QIDs
    :param sensitive_attribute: Atributos sensible
    :param insensitive_attributes: Lista de atributos no sensibles
    :param mandatory_qids: Lista de atributos obligatorios
    :param privacy_models: Lista de modelos de privacidad a aplicar
    :param output_file: Ruta del archivo CSV donde se guardan los resultados
    :param hierarchy_folder: Dirección de la carpeta que contiene los archivos {*}_hierarchy.csv
    :param max_num_of_qids: Número máximo de QIDs a considerar
    :param output_folder: Carpeta donde se exportarán los datasets anonimizados en formato .cvs.
    :param export_datasets: Si es True, se exportan los datasets anonimizados a CSV
    :param debug_comments: Si es True, se añaden comentarios de depuración al código

    :return: None
    """
    
    def getMaxHierarchyLevels(df : pd.DataFrame , path : str) -> Dict[str, int]:
        """
        Función para obtener el número total de niveles de jerarquía existentes para cada atributo.

        :param df: DataFrame con los datos originales
        :param path: Ruta donde se encuentran los archivos de jerarquía
        :return: Diccionario con los niveles de jerarquía para cada atributo
        """
        # Asegurarse de que la ruta termine con "/"
        path += "/" if path[-1] != "/" else ""

        # Diccionario para almacenar niveles de jerarquía
        hierarchy_levels = {}

        # Almacenar el número total de niveles de jerarquía existentes para cada atributo
        for atr in potential_qids:
            try:
                df_hier = pd.read_csv(f"{path}{atr}_hierarchy.csv", header=None)
                hierarchy_levels[atr] = len(df_hier.columns) - 1  # Columnas - 1 porque la primera es el nivel base
            except Exception as e:
                print(f"⚠️ No se encontró jerarquía para {atr}.")
        
        return hierarchy_levels

    def getProcessedCombinations(output_file: str) -> Tuple[pd.DataFrame, List[Set[str]]]:
        """
        Función para cargar combinaciones ya procesadas desde un archivo de salida .CSV .

        :param output_file: Ruta del archivo CSV donde se guardan los resultados
        :return: DataFrame con los resultados procesados
                    y un conjunto con las combinaciones de QIDs ya procesadas
        """
        if os.path.exists(output_file):
            processed_qids_list = []
            for row in pd.read_csv(output_file)['QIDs']:
                qids_list = row.split(", ")
                qids = {q.strip() for q in qids_list}
                processed_qids_list.append(qids)
            return pd.read_csv(output_file), processed_qids_list
        else:
            return pd.DataFrame(), []

    def calculate_risk_distance(re_id_risks : dict, 
                                keys : List[str] = ['estimated_journalist_risk', 'estimated_prosecutor_risk', 'estimated_marketer_risk']
                                ) -> float:
        """
        Calcula la distancia euclídea del perfil de riego al punto ideal cuyas coordenadas son threshold.

        :param re_id_risks: Diccionario de riesgos de re-identificación, obtenido del perfil de riesgo.
        :param keys: Claves a considerar para el cálculo de la distancia.
        :return: Distancia euclídea entre el perfil de riesgo y el punto ideal.
        """
        if any(k not in re_id_risks for k in keys):
            raise ValueError(f"Las claves {keys} no están presentes en el diccionario de riesgos de re-identificación.")
        
        risk_values = [ re_id_risks[key] for key in keys if key in keys ]
        
        return sum([r**2 for r in risk_values])**0.5

    def export_results_to_csv(df: pd.DataFrame, qids : List[str], output_folder: str) -> str:
        """
        Función para exportar el dataset anonimizado a un archivo CSV.

        :param df: DataFrame con los datos anonimizados
        :param qids: Lista de atributos QIDs utilizados en la anonimización
        :param output_folder: Carpeta donde serán exportamos los .csv anonimizados
        :return: Ruta del archivo CSV donde se guardan los resultados
        """
        # Crear el directorio de salida si no existe
        os.makedirs(output_folder, exist_ok=True)

        # Guardar el dataset anonimizado en un archivo CSV
        anon_output_path = f"{output_folder}{'_'.join(qids)}.csv"
        anonymized_df.to_csv(anon_output_path, index=False)
        return anon_output_path
    

    # Comprobar que el número de atributos obligatorios no exceda el número máximo de QIDs
    if max_num_of_qids < len(mandatory_qids):
        raise ValueError("El número máximo de QIDs no puede ser mayor que el número de atributos obligatorios.")
    elif max_num_of_qids < 1:
        raise ValueError("El número máximo de QIDs debe ser al menos 1.")
    else:
        num_of_potential_qids = max_num_of_qids - len(mandatory_qids)

    # Obtener el número total de niveles de jerarquía existentes para cada atributo
    max_hierarchy_levels = getMaxHierarchyLevels(df, hierarchy_folder) 

    # Cargar combinaciones ya procesadas
    processed_results_df, processed_qids = getProcessedCombinations(output_file)

    # Eliminamos de potential_qids los atributos obligatorios
    potential_qids = [atr for atr in potential_qids if atr not in mandatory_qids]

    # Iterar sobre todas las combinaciones de QIDs
    for qids in combinations(potential_qids, num_of_potential_qids):

        # Convertimos la tupla en lista y añadimos los atributos obligatorios
        qids = list(qids) + mandatory_qids 

        # Comprobamos si ya existe esa combinación y, en tal caso, la saltamos
        if set(qids) in processed_qids:
            if debug_comments: print(f"\n⏩ Combinación [{', '.join(qids)}] ya procesada. Saltando...")
            continue

        if debug_comments: print(f"\n🔄 Probando QIDs: {qids}")

        # Filtrar dataset con los QIDs actuales + atributo sensible
        filtered_df = df[qids + [sensitive_attribute] + insensitive_attributes]

        # Crear el Dataset
        dataset = Dataset.from_pandas(filtered_df)

        # Asignar tipo y cargar jerarquías
        for atr in qids:
            dataset.set_attribute_type(AttributeType.QUASIIDENTIFYING, atr)
            try:
                df_hier = pd.read_csv(f"{hierarchy_folder}{atr}_hierarchy.csv", header=None)
                dataset.set_hierarchy(atr, df_hier.values.tolist())
            except Exception as e:
                if debug_comments: print(f"⚠️ No se encontró jerarquía para {atr}. Saltando combinación.")
                continue
        for atr in insensitive_attributes:
            dataset.set_attribute_type(AttributeType.INSENSITIVE, atr)
        dataset.set_attribute_type(AttributeType.SENSITIVE, sensitive_attribute)

        # Intentar anonimizar
        try:
            anonymized_result = arxaas.anonymize(dataset, privacy_models)
            anonymized_df = anonymized_result.dataset.to_dataframe()
            processed_qids.append(set(qids))
            if debug_comments: print(f"✅ Anonimización de {qids} realizada")
        except Exception as e:
            if debug_comments: print(f"❌ Error en anonimización con QIDs {qids}: {e}")
            continue  # Saltar esta combinación si falla la anonimización

        # Agregar los atributos que no se han tenido en cuenta
        for atr in [col for col in df.columns if col not in qids + [sensitive_attribute] + insensitive_attributes]:
            anonymized_df[atr] = df[atr]

        anonymized_dataset = Dataset.from_pandas(anonymized_df)
        
        # Reasignar tipos en el dataset anonimizado
        for atr in qids:
            anonymized_dataset.set_attribute_type(AttributeType.QUASIIDENTIFYING, atr)
        for atr in [col for col in anonymized_df.columns if col not in qids + [sensitive_attribute]]:
            anonymized_dataset.set_attribute_type(AttributeType.INSENSITIVE, atr)
        anonymized_dataset.set_attribute_type(AttributeType.SENSITIVE, sensitive_attribute)

        # Obtener métricas de anonimización considerando todos los atributos
        try:
            anonymization_metrics_dict = anonymized_result.anonymization_metrics.attribute_generalization
        except Exception as e:
            if debug_comments: print("⚠️ No se pudieron obtener métricas de anonimización.")
            continue
        

        # Obtener perfil de riesgo
        try:
            risk_profile_original = arxaas.risk_profile(dataset)
            risk_profile_anonymized = arxaas.risk_profile(anonymized_dataset)
            re_id_risks_original = risk_profile_original.re_identification_risk
            re_id_risks_anonymized = risk_profile_anonymized.re_identification_risk
        except Exception as e:
            print("⚠️ Error al calcular perfil de riesgo. Saltando combinación.")
            continue

        # Extraer niveles de generalización en formato "actual/max"
        generalization_levels = {}
        for item in anonymization_metrics_dict:
            attr = item['name']
            if attr in max_hierarchy_levels:
                actual_level = item['generalizationLevel']
                max_level = max_hierarchy_levels[attr]
                generalization_levels[attr] = f"{actual_level}/{max_level}"

        # Claves de riesgo de re-identificación a tener en cuenta para el cálculo de distancia
        risk_keys = ['estimated_journalist_risk', 'estimated_prosecutor_risk', 'estimated_marketer_risk', 'sample_uniques']

        # Calcular la métrica de distancia
        risk_distance = calculate_risk_distance(re_id_risks_anonymized, risk_keys)
        if debug_comments: print(f"📏 Distancia de riesgo: {risk_distance:.3f}")

        # Guardar resultados con la nueva métrica
        processed_results_df = processed_results_df.append({
            'risk_distance': risk_distance,
            'QIDs': ', '.join(qids),
            **{f"{key}": f"{re_id_risks_original[key]:.3f} => {re_id_risks_anonymized[key]:.3f}" for key in risk_keys},
            'Generalization Levels': generalization_levels,
        }, ignore_index=True)
        
        # Exportar dataset anonimizado a CSV si se ha solicitado
        if export_datasets:
            anonymized_df = anonymized_df[df.columns]
            
            # Si se ha alcanzado el nivel máximo de jerarquía en algún atributo, no se incluye en el CSV   
            for atr in qids:
                if generalization_levels[atr] == f"{max_hierarchy_levels[atr]}/{max_hierarchy_levels[atr]}":
                    anonymized_df.drop(atr, axis=1, inplace=True)
                    
            try:
                res_path = export_results_to_csv(anonymized_df, qids, output_folder) 
                if debug_comments: print(f"📝 Dataset anonimizado guardado en {res_path}")
            except Exception as e:
                print(f"⚠️ Error al exportar el dataset anonimizado: {e}")

    
    # Ordenar por la métrica antes de guardar
        cols_order = ['risk_distance', 'QIDs', 'Generalization Levels'] + risk_keys
        processed_results_df = processed_results_df[cols_order]
        processed_results_df = processed_results_df.sort_values('risk_distance')
    
    # Guardar resultados en CSV
    processed_results_df.to_csv(output_file, index=False)
    print(f"\n📄 Resultados guardados en {output_file}")