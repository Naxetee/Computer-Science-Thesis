# Modelos de inteligencia artificial que preservan la privacidad

## **Introducción**

Repositorio del *Trabajo de Fin de Grado*.

- **Alumno**: *Ignacio Ávila Reyes*
- **Tutor**: *Rubén Ríos del Pozo*
- **Grado**: Ingeniería Informática
- **Centro**: *Escuela Técnica Superior de Ingeniería Informática* (ETSII), *Universidad de Málaga*
- **Curso**: *2024/2025*

## **Entorno**

Todos los notebooks de Jupyter se han ejecutado usando *Python 3.13.1*, excepto los scripts `.py` que utilizan la librería `pyarxaas`, los cuales requieren un entorno específico con *Python 3.6.0*.  
Esto se debe a que algunas librerías no soportan versiones superiores de *Python* en *Windows*.

## **Contenido**

### **./data/adults/**

Directorio que contiene los datos necesarios, organizados en subdirectorios:

- **./original/**: datos originales y preprocesados (`.csv`).
- **./hierarchies/**: jerarquías de generalización para *ARX*.
- **./results/models/**: mejores modelos obtenidos tras aplicar *GridSearchCV*.

Subdirectorios excluidos mediante `.gitignore` por su tamaño:

- **./anonymized/**: datos anonimizados con *ARX*.
- **./preprocessed/**: datos anonimizados y preprocesados para entrenamiento.
- **./results/anonymization/**: resultados de anonimización con *ARX*.
- **./results/usability/**: resultados de usabilidad de los modelos de IA.

### **./PyARXaaS_env**

Contiene un entorno virtual con Python 3.6 para instalar y usar [PyARXaaS](https://pyarxaas.readthedocs.io/en/latest/).

Este entorno está en `.gitignore` por su tamaño, pero puede crearse ejecutando:

```bash
py -3.6 -m venv PyARXaaS_env
```

Para activarlo:

```bash
./PyARXaaS_env/Scripts/activate
```

Instalar dependencias:

```bash
pip install -r PyARXaaS/requirements.txt
```

Puede ser necesario instalar `pyarxaas` desde su repositorio oficial o añadir dependencias adicionales.

### **./hierarchies**

Incluye archivos `.ipynb` que definen las jerarquías de generalización para cada conjunto de datos, necesarias para la anonimización con *ARX*.

### **./PyARXaaS**

Scripts para generar datasets anonimizados y obtener métricas de anonimización.  
Requiere un contenedor *Docker* con *ARXaaS* en ejecución. Consulta la [documentación oficial de ARXaaS](https://pyarxaas.readthedocs.io/en/latest/installation.html) para más detalles.

### **./preprocessing**

Scripts para el preprocesamiento de los datos originales (normalización, conversión de tipos, etc.), necesarios antes de entrenar los modelos de IA.

### **./machineLearning**

Scripts para seleccionar el mejor modelo de IA para los datos originales y evaluar los datos anonimizados.

### **./conclusion**

Archivos `.ipynb` con la comparativa de resultados de los modelos de anonimización y de IA.

## **Scripts**

### Anonimización

#### **./PyARXaaS/qids_selection.py**
Selecciona combinaciones de quasi-identificadores (QIDs) y ejecuta la anonimización de un dataset usando *PyARXaaS* con los modelos de privacidad especificados (*k-anonymity*, *l-diversity*, *t-closeness*). Permite definir atributos sensibles, obligatorios y no sensibles, el número máximo de QIDs, y exporta los resultados y datasets anonimizados. Es el núcleo de la selección de QIDs y la ejecución de la anonimización.

**Comando de uso:**

```bash
python PyARXaaS/qids_selection.py \
    --sensitive_attribute income \
    --mandatory_qids sex age \
    --insensitive_attributes fnlwgt \
    --max_num_of_qids 4 \
    --anonymity_models '{"k":4,"l":1,"t":0.2}' \
    --dataset_path ./data/adults/original/adults_rounded.csv \
    --output_folder ./data/adults/anonymized/ \
    --hierarchy_folder ./data/adults/hierarchies/ \
    --output_file ./data/adults/results/anonymization/4_qids/k_4-l_1-t_0.2-results.csv \
    --ARXaaS_port 8080 \
    --debug \
    --export
```

- Ajusta los argumentos según tus necesidades (atributos, rutas, modelos, etc.).
- Usa `--debug` para activar el modo depuración.
- Usa `--export` para guardar los datasets anonimizados.
#### **./PyARXaaS/anonymization_launcher.py**
Automatiza la ejecución de la anonimización de datos usando PyARXaaS para múltiples combinaciones de parámetros de privacidad (*k*, *l*, *t*) y número de QIDs. Llama internamente a `qids_selection.py`, que selecciona los quasi-identificadores, aplica los modelos de privacidad y guarda los resultados anonimizados en archivos CSV. Permite ajustar los valores de *k-anonymity*, *l-diversity*, *t-closeness* y el número máximo de QIDs, así como activar el modo depuración.

**Comando de uso:**

```bash
python PyARXaaS/anonymization_launcher.py --k 1 2 4 8 16 32 64 --l 1 2 --t 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 --q 4 --debug_comments
```

- Ajusta los valores de `--k`, `--l`, `--t` y `--q` según tus necesidades.
- Añade `--debug_comments` para ver mensajes detallados de depuración.

### Preprocesamiento

#### **./preprocessing/preprocess_original_adults.py**

Preprocesa datasets de adultos (originales o anonimizados), aplicando codificación y escalado a las variables. Detecta automáticamente columnas categóricas y numéricas, convierte la variable objetivo a binaria y guarda los datasets listos para machine learning en la carpeta de salida.

**Comando:**

```bash
python preprocessing/anonymized_adults_preprocessing.py --debug_comments --isOriginal
```

- Usa `--isOriginal` para procesar el dataset original; si lo omites, procesa los anonimizados.
- Añade `--debug_comments` para ver mensajes detallados.

### Machine Learning

#### **./machineLearning/gridSearch_original_adults.py**

Entrena varios modelos de machine learning (Logistic Regression, KNN, Random Forest, XGBoost) sobre el dataset original usando *GridSearchCV* para encontrar los mejores hiperparámetros. Selecciona el mejor modelo según la precisión y guarda sus parámetros y métricas.

**Comando de ejemplo:**

```bash
python machineLearning/gridSearch_original_adults.py --input_path <ruta_al_csv> --output_path <ruta_salida_csv> --save_results --debug_comments
```

#### **./machineLearning/comparing_anonymized.py**

Evalúa el mejor modelo obtenido sobre el dataset original en múltiples datasets anonimizados. Calcula métricas de accuracy, precision, recall y f1-score para cada dataset y guarda los resultados en un archivo CSV.

**Comando:**

```bash
python machineLearning/comparing_anonymized.py --model_path <ruta_modelo_csv> --datasets_path <carpeta_datasets> --original_dataset_path <ruta_original_csv> --output_path <ruta_salida_csv> --debug_comments
```

#### **./machineLearning/usability_metrics_from_generalization_fractions.py**

Reconstruye datasets anonimizados a partir de los niveles de generalización aplicados, usando las jerarquías de atributos. Evalúa la usabilidad de los datos anonimizados aplicando el mejor modelo y calcula métricas de utilidad (accuracy, precision, recall, f1-score) para cada combinación de QIDs y niveles de generalización.

**Comando:**

```bash
python machineLearning/usability_metrics_from_generalization_fractions.py --anon_results_path <carpeta_anonimizacion> --usability_results_path <carpeta_usabilidad> --hier_path <carpeta_jerarquias> --original_dataset_path <ruta_original_csv> --best_model_path <ruta_modelo_csv> --q <num_qids> --debug_comments
```
