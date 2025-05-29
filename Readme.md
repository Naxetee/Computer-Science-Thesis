# Modelos de inteligencia artificial que preservan la privacidad.
## **Introducción**
Repositorio de *Trabajo de Fin de Grado*.
- **Alumno**: *Ignacio Ávila Reyes*
- **Tutor**: *Rubén Ríos del Pozo*
- **Grado**: Ingeniería Informática
- **Centro**: *Escuela Técnica Superior de Ingeniería Informática* (ETSII) de la *Universidad de Málaga *
- **Curso**: *2024/2025*

## **Entorno**
Todos los notebooks de Jupyter se han ejecutado usando *`Python 3.9.13`* excepto aquellos scripts `.py` que usan la librería `pyarxaas`, que han sido ejecutados desde un ambiente de *`Python 3.6.0`* creado específicamente para ello.

Esto se debe a que estas librerías no soportan una versión de *`Python`* superior a la *`3.6.0`* en *Windows*.

## **Contenido**
#### **./data**
Este directorio alberga todos los datos necesarios referentes a cada fuente de datos, con su respectivo nombre en cada subdirectorio. 

Dentro de estos, tenemos archivos en formato `.csv` que contienen:
- los datos originales.
- los datos anonimizados.
- las *[jerarquías de generalización]()* definidas para el uso de *ARX*. 
- algunos datos auxiliares con cambios de formato.
- ...

#### **./hierarchies**

Contiene los notebooks de Jupyter en los que definimos manualmente las *[jerarquías de generalización]()* usadas para anonimizar cada atributo de cada dataset con *ARX*.

#### **./preprocessing**
En la carpeta *[preprocessing]()* encontramos los notebooks de Jupyter en los que cargamos por primera vez los datasets originales y los preparamos para su posterior uso.

Los datos ya preparados se guardan en *[./data/'dataset'/'dataset'_orinial.csv]()*

#### **./PyARXaaS**
Contiene scripts `'dataset'_pyarxaas.py`, que anonimizan datasets usando *k-anonymity*, *l-diversity* y *t-closeness* y los guardan en *[./data/'dataset'/anonymized]()*

#### **./PyARXaaS_env**
Contiene un entorno con Python 3.6, necesario para instalar y usar *[PyARXaaS]()*. Se puede activar ejecutando `./PyARXaaS_env/Scripts/activate` y una vez activado podemos probar el script `pyarxaas_adults.py`.

#### **./testing**
Contiene algunos archivos que han sido creados con la única finalidad de realizar pruebas sobre diversas herramientas antes de incluir su uso en los notebooks y scripts importantes.