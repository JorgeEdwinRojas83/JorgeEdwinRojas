import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import streamlit as st
from datetime import datetime
import json
import os
from typing import List, Dict, Tuple, Optional, Any


# ============================================================================
# FUNCIONES DE CONFIGURACIÓN Y CARGA
# ============================================================================

def cargar_modelo_nlp():
    """Cargar modelo de spaCy"""
    modelo_nlp = spacy.load("es_core_news_md")
    if "sentencizer" not in modelo_nlp.pipe_names:
        modelo_nlp.add_pipe("sentencizer", before="parser")

    # ✅ Agrega la regla de puntuación una sola vez
    modelo_nlp.get_pipe("sentencizer").punct_chars.add('.')
    # Configurar casos especiales
    casos_especiales = ["1.", "2.", "ej.", "etc."]
    for caso in casos_especiales:
        modelo_nlp.tokenizer.add_special_case(caso, [{"ORTH": caso}])
    return modelo_nlp


def cargar_modelo_embeddings():
    """Cargar modelo de embeddings"""
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


# ============================================================================
# FUNCIONES DE PROCESAMIENTO DE TEXTO
# ============================================================================

def limpiar_texto(texto: str) -> str:
    """Limpiar y normalizar texto"""
    if not isinstance(texto, str):
        return ""

    # Limpiar espacios en blanco repetidos
    texto = re.sub(r'\s+', ' ', texto).strip()
    texto = re.sub(r'\s\.', '. ', texto)  # cambia " . " por ". "
    texto = re.sub(r'eg\.\s', 'eg, ', texto)  # cambia "eg. " por "eg, "
    texto = re.sub(r'etc\.\s', 'etc, ', texto)  # cambia "etc. " por "etc, "
    texto = texto.replace(".,", ",")  # cambia "., " por ", "
    texto = texto.replace("..", ".")  # cambia ".." por "."
    return texto


def dividir_texto_en_oraciones(texto: str, modelo_nlp) -> List[str]:
    """Dividir texto en oraciones usando spaCy"""
    texto_limpio = limpiar_texto(texto)
    if not texto_limpio:
        return []

    documento = modelo_nlp(texto_limpio)
    oraciones = []
    for oracion in documento.sents:
        oracion_limpia = oracion.text.strip()
        if oracion_limpia:
            oraciones.append(oracion_limpia)

    return oraciones


# ============================================================================
# FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================

def cargar_archivo_excel(ruta_archivo: str) -> Tuple[pd.DataFrame, str]:
    """Cargar archivo Excel y extraer información básica"""
    try:
        dataframe = pd.read_excel(ruta_archivo)
        if len(dataframe.columns) == 0:
            raise Exception("El archivo no tiene columnas")

        primera_columna = dataframe.columns[0]
        titulo_pregunta = primera_columna

        print(f"Archivo cargado: {len(dataframe)} filas")
        print(f"Pregunta detectada: {titulo_pregunta}")

        return dataframe, titulo_pregunta

    except Exception as e:
        raise Exception(f"Error al cargar archivo: {str(e)}")


def extraer_respuestas_validas(dataframe: pd.DataFrame, nombre_columna: str) -> pd.DataFrame:
    """Extraer respuestas válidas (no nulas)"""
    dataframe_limpio = dataframe.dropna(subset=[nombre_columna])
    dataframe_limpio = dataframe_limpio[dataframe_limpio[nombre_columna].str.strip() != '']
    print(f"Respuestas válidas: {len(dataframe_limpio)}")
    return dataframe_limpio


def crear_datos_oraciones(dataframe_respuestas: pd.DataFrame, nombre_columna: str, modelo_nlp) -> List[Dict]:
    """Crear estructura de datos para todas las oraciones"""
    todas_las_oraciones = []
    contador_oraciones = 0

    for indice, fila in dataframe_respuestas.iterrows():
        id_respuesta = f"R{indice + 1}"
        texto_completo = str(fila[nombre_columna])

        if texto_completo.strip():
            oraciones = dividir_texto_en_oraciones(texto_completo, modelo_nlp)

            for numero_oracion, texto_oracion in enumerate(oraciones):
                datos_oracion = {
                    'id': f"{id_respuesta}_s{numero_oracion + 1}",
                    'id_respuesta': id_respuesta,
                    'numero_oracion': numero_oracion + 1,
                    'texto': texto_oracion,
                    'respuesta_completa': texto_completo,
                    'numero_fila': indice + 1,
                    'indice_global': contador_oraciones
                }
                todas_las_oraciones.append(datos_oracion)
                contador_oraciones += 1

    print(f"Total de oraciones procesadas: {len(todas_las_oraciones)}")
    return todas_las_oraciones


# ============================================================================
# FUNCIONES DE SIMILITUD
# ============================================================================

def calcular_matriz_similitud(oraciones: List[Dict], modelo_oraciones) -> np.ndarray:
    """Calcular matriz de similitud semántica"""
    if not oraciones:
        return np.array([])

    textos = [oracion['texto'] for oracion in oraciones]
    embeddings = modelo_oraciones.encode(textos, convert_to_tensor=True)
    matriz_similitud = cosine_similarity(embeddings)

    print("Matriz de similitud calculada")
    return matriz_similitud


def encontrar_oraciones_similares(indice_oracion: int, matriz_similitud: np.ndarray,
                                  todas_las_oraciones: List[Dict], umbral: float = 0.8) -> List[Dict]:
    """Encontrar oraciones similares a una dada"""
    if matriz_similitud.size == 0:
        return []

    lista_similares = []
    for i in range(len(todas_las_oraciones)):
        if i != indice_oracion:
            puntaje_similitud = matriz_similitud[indice_oracion][i]
            if puntaje_similitud > umbral:
                lista_similares.append({
                    'indice': i,
                    'datos_oracion': todas_las_oraciones[i],
                    'similitud': puntaje_similitud
                })

    # Ordenar por similitud descendente
    lista_similares.sort(key=lambda x: x['similitud'], reverse=True)
    return lista_similares


# ============================================================================
# FUNCIONES DE AGRUPACIÓN Y ORGANIZACIÓN
# ============================================================================

def agrupar_oraciones_por_respuesta(oraciones: List[Dict]) -> Dict[str, Dict]:
    """Organizar oraciones por ID de respuesta"""
    respuestas_agrupadas = {}

    for oracion in oraciones:
        id_respuesta = oracion['id_respuesta']
        if id_respuesta not in respuestas_agrupadas:
            respuestas_agrupadas[id_respuesta] = {
                'oraciones': [],
                'informacion': {
                    'numero_fila': oracion['numero_fila'],
                    'longitud_texto_total': len(oracion['respuesta_completa'])
                }
            }
        respuestas_agrupadas[id_respuesta]['oraciones'].append(oracion)

    return respuestas_agrupadas


def calcular_estadisticas_respuesta(datos_respuesta: Dict) -> Dict[str, int]:
    """Calcular estadísticas de una respuesta"""
    oraciones = datos_respuesta['oraciones']
    if not oraciones:
        return {'total_palabras': 0, 'promedio_palabras_por_oracion': 0}

    total_palabras = sum(len(oracion['texto'].split()) for oracion in oraciones)
    promedio_palabras = round(total_palabras / len(oraciones))

    return {
        'total_palabras': total_palabras,
        'promedio_palabras_por_oracion': promedio_palabras
    }


# ============================================================================
# FUNCIONES DE GESTIÓN DE RETROALIMENTACIÓN
# ============================================================================

def cargar_datos_retroalimentacion(archivo_retroalimentacion: str = "datos_retroalimentacion.json") -> Dict:
    """Cargar retroalimentación guardada"""
    if os.path.exists(archivo_retroalimentacion):
        try:
            with open(archivo_retroalimentacion, 'r', encoding='utf-8') as archivo:
                return json.load(archivo)
        except:
            return {}
    return {}


def guardar_datos_retroalimentacion(datos_retroalimentacion: Dict,
                                    archivo_retroalimentacion: str = "datos_retroalimentacion.json") -> bool:
    """Guardar retroalimentación en archivo"""
    try:
        with open(archivo_retroalimentacion, 'w', encoding='utf-8') as archivo:
            json.dump(datos_retroalimentacion, archivo, ensure_ascii=False, indent=2)
        return True
    except Exception as error:
        st.error(f"Error guardando: {str(error)}")
        return False


def crear_objeto_retroalimentacion(comentario: str, es_generalizada: bool = False,
                                   oracion_fuente: str = None) -> Dict:
    """Crear objeto de retroalimentación (sin calificación)"""
    return {
        'comentario': comentario,
        'fecha': datetime.now().isoformat(),
        'generalizada': es_generalizada,
        'oracion_fuente': oracion_fuente
    }


def guardar_retroalimentacion_individual(datos_retroalimentacion: Dict, id_respuesta: str,
                                         id_oracion: str, comentario: str) -> Dict:
    """Guardar retroalimentación individual para una oración (sin calificación)"""
    if id_respuesta not in datos_retroalimentacion:
        datos_retroalimentacion[id_respuesta] = {}
    if 'oraciones' not in datos_retroalimentacion[id_respuesta]:
        datos_retroalimentacion[id_respuesta]['oraciones'] = {}

    objeto_retroalimentacion = crear_objeto_retroalimentacion(comentario, False)
    datos_retroalimentacion[id_respuesta]['oraciones'][id_oracion] = objeto_retroalimentacion

    return datos_retroalimentacion


def aplicar_retroalimentacion_generalizada(datos_retroalimentacion: Dict, oracion_fuente: Dict,
                                           comentario: str,
                                           oraciones_similares: List[Dict]) -> Tuple[Dict, int]:
    """Aplicar retroalimentación generalizada a oraciones similares (sin calificación)"""
    id_fuente = oracion_fuente['id']
    respuesta_fuente = oracion_fuente['id_respuesta']

    # Crear objeto de retroalimentación
    objeto_retroalimentacion = crear_objeto_retroalimentacion(comentario, True, id_fuente)

    # Guardar en oración fuente
    if respuesta_fuente not in datos_retroalimentacion:
        datos_retroalimentacion[respuesta_fuente] = {}
    if 'oraciones' not in datos_retroalimentacion[respuesta_fuente]:
        datos_retroalimentacion[respuesta_fuente]['oraciones'] = {}

    datos_retroalimentacion[respuesta_fuente]['oraciones'][id_fuente] = objeto_retroalimentacion

    # Aplicar a oraciones similares
    contador_aplicadas = 1  # Contar la fuente

    for similar in oraciones_similares:
        oracion_similar = similar['datos_oracion']
        respuesta_similar = oracion_similar['id_respuesta']
        id_similar = oracion_similar['id']

        # Crear estructura si no existe
        if respuesta_similar not in datos_retroalimentacion:
            datos_retroalimentacion[respuesta_similar] = {}
        if 'oraciones' not in datos_retroalimentacion[respuesta_similar]:
            datos_retroalimentacion[respuesta_similar]['oraciones'] = {}

        # Crear copia de la retroalimentación con información adicional
        retroalimentacion_similar = objeto_retroalimentacion.copy()
        retroalimentacion_similar['oracion_objetivo'] = id_similar
        datos_retroalimentacion[respuesta_similar]['oraciones'][id_similar] = retroalimentacion_similar
        contador_aplicadas += 1

    return datos_retroalimentacion, contador_aplicadas


def buscar_retroalimentacion_existente(id_oracion: str, todos_los_datos_retroalimentacion: Dict) -> Optional[Dict]:
    """Buscar retroalimentación existente para una oración"""
    for id_respuesta, datos_respuesta in todos_los_datos_retroalimentacion.items():
        if 'oraciones' in datos_respuesta:
            if id_oracion in datos_respuesta['oraciones']:
                return datos_respuesta['oraciones'][id_oracion]
    return None


# ============================================================================
# FUNCIONES DE DASHBOARD DE RESULTADOS SIMPLIFICADO
# ============================================================================

def crear_dataframe_dashboard_simplificado() -> pd.DataFrame:
    """Crear DataFrame completo con todos los resultados de evaluación (sin calificaciones)"""
    if 'todas_las_oraciones' not in st.session_state:
        return pd.DataFrame()

    filas_dashboard = []

    for indice, oracion in enumerate(st.session_state.todas_las_oraciones):
        retroalimentacion_oracion = buscar_retroalimentacion_existente(
            oracion['id'], st.session_state.datos_retroalimentacion
        )

        # Estados básicos
        estado_evaluacion = "Sin evaluar"
        tipo_aplicacion = "Manual"
        comentario_texto = ""
        fecha_evaluacion = ""
        oracion_fuente_id = ""

        if retroalimentacion_oracion:
            estado_evaluacion = "Evaluado"
            comentario_texto = retroalimentacion_oracion.get('comentario', '')
            fecha_evaluacion = retroalimentacion_oracion.get('fecha', '')

            if retroalimentacion_oracion.get('generalizada'):
                if retroalimentacion_oracion.get('oracion_fuente') == oracion['id']:
                    tipo_aplicacion = "Fuente generalización"
                else:
                    tipo_aplicacion = "Generalizado"

            oracion_fuente_id = retroalimentacion_oracion.get('oracion_fuente', '')

        # Estadísticas de texto
        num_palabras = len(oracion['texto'].split())
        longitud_caracteres = len(oracion['texto'])

        fila_dashboard = {
            'Numero_Global': indice + 1,
            'ID_Respuesta': oracion['id_respuesta'],
            'Numero_Oracion': oracion['numero_oracion'],
            'ID_Oracion_Completo': oracion['id'],
            'Fila_Original': oracion['numero_fila'],
            'Texto_Oracion': oracion['texto'],
            'Palabras': num_palabras,
            'Caracteres': longitud_caracteres,
            'Estado': estado_evaluacion,
            'Tipo_Aplicacion': tipo_aplicacion,
            'Comentario': comentario_texto,
            'Fecha': fecha_evaluacion[:10] if fecha_evaluacion else "",  # Solo fecha, sin hora
            'Oracion_Fuente': oracion_fuente_id
        }
        filas_dashboard.append(fila_dashboard)

    return pd.DataFrame(filas_dashboard)


def mostrar_estadisticas_dashboard_simplificado(df_dashboard: pd.DataFrame) -> None:
    """Mostrar estadísticas generales del dashboard (sin calificaciones)"""
    if df_dashboard.empty:
        st.warning("No hay datos para mostrar estadísticas")
        return

    total_oraciones = len(df_dashboard)
    evaluadas = len(df_dashboard[df_dashboard['Estado'] == 'Evaluado'])
    sin_evaluar = total_oraciones - evaluadas

    # Contadores por tipo
    manuales = len(df_dashboard[df_dashboard['Tipo_Aplicacion'] == 'Manual'])
    generalizadas = len(df_dashboard[df_dashboard['Tipo_Aplicacion'] == 'Generalizado'])
    fuentes = len(df_dashboard[df_dashboard['Tipo_Aplicacion'] == 'Fuente generalización'])

    st.subheader("Estadísticas Generales")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Oraciones", total_oraciones)
        st.metric("Sin Evaluar", sin_evaluar)

    with col2:
        st.metric("Evaluadas", evaluadas, delta=f"{(evaluadas / total_oraciones * 100):.1f}%")
        st.metric("Manuales", manuales)

    with col3:
        st.metric("Generalizadas", generalizadas)
        st.metric("Fuentes", fuentes)

    with col4:
        # Mostrar porcentaje de generalizadas
        porcentaje_gen = (generalizadas / evaluadas * 100) if evaluadas > 0 else 0
        st.metric("% Generalizadas", f"{porcentaje_gen:.1f}%")


def exportar_dashboard_tabla_completa(df_dashboard: pd.DataFrame) -> Tuple[str, str]:
    """Exportar tabla completa del dashboard a Excel y JSON"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Exportar a Excel
        nombre_archivo_excel = f"tabla_resultados_completa_{timestamp}.xlsx"
        df_dashboard.to_excel(nombre_archivo_excel, index=False)

        # Exportar a JSON
        nombre_archivo_json = f"tabla_resultados_completa_{timestamp}.json"
        df_dashboard.to_json(nombre_archivo_json, orient='records', indent=2, force_ascii=False)

        return nombre_archivo_excel, nombre_archivo_json

    except Exception as error:
        st.error(f"Error en exportación: {str(error)}")
        return None, None


def mostrar_dashboard_simplificado() -> None:
    """Mostrar dashboard simplificado de resultados (sin filtros)"""
    st.header("Dashboard de Resultados")

    if 'todas_las_oraciones' not in st.session_state:
        st.warning("No hay datos cargados. Ve a la pestaña principal y carga un archivo Excel.")
        return

    # Crear DataFrame completo
    df_dashboard = crear_dataframe_dashboard_simplificado()

    if df_dashboard.empty:
        st.error("Error creando el dashboard. Verifica que los datos estén cargados correctamente.")
        return

    # Mostrar estadísticas generales
    mostrar_estadisticas_dashboard_simplificado(df_dashboard)

    st.divider()

    # Opciones de visualización simples
    st.subheader("Opciones de Visualización")
    col1, col2 = st.columns(2)

    with col1:
        mostrar_texto_completo = st.checkbox("Mostrar texto completo", value=False)

    with col2:
        if st.button("Exportar tabla completa (Excel + JSON)"):
            archivo_excel, archivo_json = exportar_dashboard_tabla_completa(df_dashboard)
            if archivo_excel and archivo_json:
                st.success(f"Exportado Excel: {archivo_excel}")
                st.success(f"Exportado JSON: {archivo_json}")

    # Preparar columnas para mostrar
    columnas_mostrar = [
        'Numero_Global', 'ID_Respuesta', 'Numero_Oracion', 'Estado',
        'Tipo_Aplicacion', 'Palabras'
    ]

    if mostrar_texto_completo:
        columnas_mostrar.extend(['Texto_Oracion', 'Comentario'])
    else:
        # Crear previews
        df_dashboard_mostrar = df_dashboard.copy()
        df_dashboard_mostrar['Texto_Preview'] = df_dashboard['Texto_Oracion'].apply(
            lambda x: x[:60] + "..." if len(x) > 60 else x
        )
        df_dashboard_mostrar['Comentario_Preview'] = df_dashboard['Comentario'].apply(
            lambda x: (x[:40] + "..." if len(x) > 40 else x) if x else ""
        )
        columnas_mostrar.extend(['Texto_Preview', 'Comentario_Preview'])
        df_dashboard = df_dashboard_mostrar

    # Mostrar información
    st.info(f"Mostrando todas las {len(df_dashboard)} oraciones")

    # Mostrar tabla completa (sin paginación)
    st.subheader("Tabla de Resultados Completa")

    if len(df_dashboard) > 0:
        # Mostrar tabla sin colores
        st.dataframe(df_dashboard[columnas_mostrar], use_container_width=True, height=600)

        # Leyenda de estados
        st.markdown("""
        **Estados de evaluación:**
        - Sin evaluar | Evaluado  
        **Tipos de aplicación:**
        - Manual | Generalizado | Fuente generalización
        """)

        # Gráficos de análisis simples
        st.subheader("Análisis Visual")
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de estados
            estados_count = df_dashboard['Estado'].value_counts()
            st.bar_chart(estados_count, height=300)
            st.caption("Estados de Evaluación")

        with col2:
            # Gráfico de tipos de aplicación
            tipos_count = df_dashboard['Tipo_Aplicacion'].value_counts()
            st.bar_chart(tipos_count, height=300)
            st.caption("Tipos de Aplicación")

    else:
        st.warning("No hay datos para mostrar.")


# ============================================================================
# FUNCIONES DE INTERFAZ DE USUARIO
# ============================================================================

def renderizar_caja_oracion(oracion: Dict, es_similar: bool, estado_generalizacion: str) -> None:
    """Renderizar caja de oración con estilos apropiados"""
    clase_css = "caja-oracion"
    if es_similar:
        clase_css += " oracion-similar"

    indicador_generalizacion = ""
    if estado_generalizacion == "fuente":
        indicador_generalizacion = ' <span style="color: #28a745; font-size: 11px;">🎯 FUENTE</span>'
    elif estado_generalizacion == "heredada":
        indicador_generalizacion = ' <span style="color: #6c757d; font-size: 11px;">🔄 HEREDADA</span>'

    st.markdown(f"""
    <div class="{clase_css}">
        <div class="texto-oracion">
            <strong>Oración {oracion['numero_oracion']}:</strong>{indicador_generalizacion}<br>
            {oracion['texto']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def renderizar_informacion_oraciones_similares(oraciones_similares: List[Dict],
                                               todos_los_datos_retroalimentacion: Dict) -> None:
    """Renderizar información de oraciones similares"""
    st.markdown("**🔗 Oraciones que recibirán esta retroalimentación:**")

    for similar in oraciones_similares:
        retroalimentacion_existente = buscar_retroalimentacion_existente(
            similar['datos_oracion']['id'], todos_los_datos_retroalimentacion
        )
        icono_estado = "✅" if retroalimentacion_existente else "⏳"
        vista_previa_oracion = (similar['datos_oracion']['texto']
                                if len(similar['datos_oracion']['texto']) > 50
                                else similar['datos_oracion']['texto'])
        st.write(f"{icono_estado} **{similar['datos_oracion']['id_respuesta']}**: {vista_previa_oracion}")


def renderizar_formulario_retroalimentacion(oracion: Dict, retroalimentacion_actual: Dict,
                                                         oraciones_similares: List[Dict],
                                                         todos_los_datos_retroalimentacion: Dict) -> str:
    """Renderizar formulario de retroalimentación simplificado (solo comentario)"""
    # Mostrar información de generalización si existe
    if retroalimentacion_actual.get('generalizada'):
        if retroalimentacion_actual.get('oracion_fuente') == oracion['id']:
            st.info("🎯 Esta oración fue la fuente de generalización")
        else:
            id_fuente = retroalimentacion_actual.get('oracion_fuente', 'desconocida')
            st.info(f"🔄 Heredado manualmente de: {id_fuente}")

    # Mostrar oraciones similares si las hay
    if oraciones_similares:
        renderizar_informacion_oraciones_similares(oraciones_similares, todos_los_datos_retroalimentacion)
        st.markdown("---")

    # Formulario de retroalimentación (solo comentario)
    comentario = st.text_area(
        "Comentario:",
        value=retroalimentacion_actual.get('comentario', ''),
        key=f"comentario_{oracion['id']}",
        height=100,
        help="Escribir retroalimentación para esta oración"
    )

    return comentario


def inicializar_estado_sesion() -> None:
    """Inicializar variables de estado de sesión"""
    if 'modelo_nlp' not in st.session_state:
        st.session_state.modelo_nlp = cargar_modelo_nlp()

    if 'modelo_oraciones' not in st.session_state:
        st.session_state.modelo_oraciones = cargar_modelo_embeddings()

    # Configuración por defecto para umbral único
    if 'umbral_similitud' not in st.session_state:
        st.session_state.umbral_similitud = 0.80


def procesar_archivo_subido(archivo_subido) -> bool:
    """Procesar archivo subido y actualizar estado de sesión"""
    ruta_temporal = "respuestas_temporales.xlsx"
    with open(ruta_temporal, "wb") as archivo:
        archivo.write(archivo_subido.getbuffer())

    nombre_archivo_actual = archivo_subido.name

    # Verificar si es un archivo nuevo
    if ('archivo_actual' not in st.session_state or
            st.session_state.archivo_actual != nombre_archivo_actual):

        try:
            # Cargar y procesar datos
            dataframe, titulo_pregunta = cargar_archivo_excel(ruta_temporal)
            dataframe_limpio = extraer_respuestas_validas(dataframe, dataframe.columns[0])
            todas_las_oraciones = crear_datos_oraciones(
                dataframe_limpio, dataframe.columns[0], st.session_state.modelo_nlp
            )
            matriz_similitud = calcular_matriz_similitud(todas_las_oraciones, st.session_state.modelo_oraciones)
            respuestas_agrupadas = agrupar_oraciones_por_respuesta(todas_las_oraciones)
            datos_retroalimentacion = cargar_datos_retroalimentacion()

            # Actualizar estado de sesión
            st.session_state.archivo_actual = nombre_archivo_actual
            st.session_state.dataframe_original = dataframe
            st.session_state.titulo_pregunta = titulo_pregunta
            st.session_state.todas_las_oraciones = todas_las_oraciones
            st.session_state.matriz_similitud = matriz_similitud
            st.session_state.respuestas_agrupadas = respuestas_agrupadas
            st.session_state.datos_retroalimentacion = datos_retroalimentacion

            st.success(f"📄 Nuevo archivo cargado: {nombre_archivo_actual}")
            return True

        except Exception as error:
            st.error(f"Error procesando archivo: {str(error)}")
            return False

    return True


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def principal():
    st.set_page_config(
        page_title="Sistema de Retroalimentación Simplificado",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Sistema de Retroalimentación Académica")
    st.subheader("Análisis automático de respuestas estudiantiles con IA")

    # Inicializar estado de sesión
    inicializar_estado_sesion()

    # Crear pestañas principales
    tab1, tab2 = st.tabs(["🏠 Evaluación Principal", "📊 Dashboard de Resultados"])

    with tab1:
        mostrar_pestana_evaluacion()

    with tab2:
        mostrar_dashboard_simplificado()


def mostrar_pestana_evaluacion():
    """Mostrar la pestaña principal de evaluación simplificada"""

    # Estilos CSS
    st.markdown("""
    <style>
    .caja-oracion {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        border-left: 4px solid #007bff;
    }

    .oracion-similar {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }

    .texto-oracion {
        color: #212529;
        font-size: 14px;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)

    # Panel lateral de configuración
    with st.sidebar:
        st.header("⚙️ Configuración")

        archivo_subido = st.file_uploader(
            "Subir archivo Excel",
            type=['xlsx', 'xls'],
            help="Archivo con respuestas de Google Forms"
        )

        if archivo_subido is not None:
            if procesar_archivo_subido(archivo_subido):
                st.info(f"📊 {len(st.session_state.todas_las_oraciones)} oraciones encontradas")
                st.info(f"📁 Archivo actual: {st.session_state.archivo_actual}")

                if st.session_state.titulo_pregunta:
                    st.markdown(f"**❓ Pregunta:** {st.session_state.titulo_pregunta}")

                # Estadísticas
                total_respuestas = len([oracion for oracion in st.session_state.todas_las_oraciones
                                        if oracion['numero_oracion'] == 1])
                st.metric("Total respuestas", total_respuestas)
                st.metric("Respuestas válidas", len(st.session_state.respuestas_agrupadas))

        # Configuraciones simplificadas (un solo umbral)
        st.subheader("🔧 Parámetros")
        st.session_state.umbral_similitud = st.slider(
            "Umbral de similitud",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.umbral_similitud,
            step=0.01,
            help="Umbral único para similitud de oraciones y comentarios"
        )

        mostrar_similitudes = st.checkbox("Mostrar similitudes", value=True)
        retroalimentacion_oracion_habilitada = st.checkbox("Retroalimentación por oración", value=True)

        # Botón de limpieza
        st.markdown("---")
        if st.button("🗑️ Limpiar Todo", help="Elimina todos los datos cargados"):
            for clave in list(st.session_state.keys()):
                if clave not in ['modelo_nlp', 'modelo_oraciones']:  # Mantener modelos cargados
                    del st.session_state[clave]
            st.success("✅ Datos limpiados")
            st.rerun()

        # Exportación
        st.subheader("📤 Exportación")

        if st.button("📋 Exportar Oraciones (Excel + JSON)"):
            if 'todas_las_oraciones' in st.session_state:
                # Crear DataFrame del dashboard y exportar en ambos formatos
                df_dashboard = crear_dataframe_dashboard_simplificado()
                if not df_dashboard.empty:
                    archivo_excel, archivo_json = exportar_dashboard_tabla_completa(df_dashboard)
                    if archivo_excel and archivo_json:
                        st.success(f"✅ Exportado Excel: {archivo_excel}")
                        st.success(f"✅ Exportado JSON: {archivo_json}")
                else:
                    st.error("Error creando los datos para exportar")

    # Contenido principal
    if 'respuestas_agrupadas' in st.session_state:
        respuestas = st.session_state.respuestas_agrupadas

        if respuestas:
            # Selector de respuesta
            columna1, columna2, columna3 = st.columns([2, 1, 1])

            with columna1:
                respuesta_seleccionada = st.selectbox(
                    "Seleccionar respuesta:",
                    options=list(respuestas.keys()),
                    format_func=lambda x: f"{x} ({len(respuestas[x]['oraciones'])} oraciones)"
                )

            with columna2:
                contador_evaluadas = 0
                for r in respuestas.keys():
                    if st.session_state.datos_retroalimentacion.get(r, {}).get('oraciones'):
                        contador_evaluadas += 1
                st.metric("Evaluadas", f"{contador_evaluadas}/{len(respuestas)}")

            with columna3:
                total_oraciones = sum(len(r['oraciones']) for r in respuestas.values())
                # Contar oraciones con retroalimentación generalizada
                total_generalizadas = 0
                for r in respuestas.values():
                    for oracion in r['oraciones']:
                        retroalimentacion = buscar_retroalimentacion_existente(
                            oracion['id'], st.session_state.datos_retroalimentacion
                        )
                        if retroalimentacion and retroalimentacion.get('generalizada'):
                            total_generalizadas += 1
                st.metric("Generalizadas", f"{total_generalizadas}/{total_oraciones}")

            # Mostrar respuesta seleccionada
            if respuesta_seleccionada:
                datos_respuesta = respuestas[respuesta_seleccionada]
                st.header(f"🎓 Evaluando {respuesta_seleccionada}")
                st.text(datos_respuesta['oraciones'][0]['respuesta_completa'])

                # Información básica
                with st.expander("ℹ️ Información de la respuesta", expanded=True):
                    columna1, columna2, columna3, columna4 = st.columns(4)

                    estadisticas = calcular_estadisticas_respuesta(datos_respuesta)

                    with columna1:
                        st.metric("Oraciones", len(datos_respuesta['oraciones']))
                    with columna2:
                        st.metric("Fila original", datos_respuesta['informacion']['numero_fila'])
                    with columna3:
                        st.metric("Longitud texto (caracteres)",
                                  f"{datos_respuesta['informacion']['longitud_texto_total']} ")
                    with columna4:
                        st.metric("Promedio palabras/oración", estadisticas['promedio_palabras_por_oracion'])

                # Mostrar oraciones
                st.subheader("📝 Oraciones")

                retroalimentacion_actual = st.session_state.datos_retroalimentacion.get(respuesta_seleccionada, {})
                retroalimentacion_oraciones = retroalimentacion_actual.get('oraciones', {})

                for oracion in datos_respuesta['oraciones']:
                    # Encontrar similitudes usando el umbral único
                    oraciones_similares = []
                    if mostrar_similitudes:
                        indice_oracion = oracion['indice_global']
                        oraciones_similares = encontrar_oraciones_similares(
                            indice_oracion,
                            st.session_state.matriz_similitud,
                            st.session_state.todas_las_oraciones,
                            st.session_state.umbral_similitud
                        )

                    # Determinar estado de generalización
                    retroalimentacion_oracion_actual = retroalimentacion_oraciones.get(oracion['id'], {})
                    estado_generalizacion = "normal"

                    if retroalimentacion_oracion_actual.get('generalizada'):
                        if retroalimentacion_oracion_actual.get('oracion_fuente') == oracion['id']:
                            estado_generalizacion = "fuente"
                        else:
                            estado_generalizacion = "heredada"

                    # Renderizar oración
                    renderizar_caja_oracion(oracion, bool(oraciones_similares), estado_generalizacion)

                    # Panel de similitudes
                    if oraciones_similares and mostrar_similitudes:
                        with st.expander(f"⚠️ {len(oraciones_similares)} oraciones similares"):
                            for similar in oraciones_similares:
                                existente = buscar_retroalimentacion_existente(
                                    similar['datos_oracion']['id'], st.session_state.datos_retroalimentacion
                                )
                                estado = "✅ Evaluada" if existente else "⏳ Pendiente"

                                st.write(
                                    f"**{similar['datos_oracion']['id_respuesta']}-s{similar['datos_oracion']['numero_oracion']}** "
                                    f"(Similitud: {similar['similitud']:.1%}) - {estado}")
                                #st.write(f"_{similar['datos_oracion']['texto'][:80]}..._")
                                st.write(f"_{similar['datos_oracion']['texto']}_")

                    # Panel de retroalimentación simplificada
                    if retroalimentacion_oracion_habilitada:
                        titulo_expandir = f"💬 Retroalimentación - Oración {oracion['numero_oracion']}"
                        if estado_generalizacion == "fuente":
                            titulo_expandir += " (🎯)"
                        elif estado_generalizacion == "heredada":
                            titulo_expandir += " (🔄)"

                        with st.expander(titulo_expandir):
                            comentario = renderizar_formulario_retroalimentacion(
                                oracion, retroalimentacion_oracion_actual, oraciones_similares,
                                st.session_state.datos_retroalimentacion
                            )

                            # Botones de acción simplificados
                            columna_btn1, columna_btn2 = st.columns(2)

                            with columna_btn1:
                                if st.button("💾 Solo Esta", key=f"individual_{oracion['id']}"):
                                    st.session_state.datos_retroalimentacion = guardar_retroalimentacion_individual(
                                        st.session_state.datos_retroalimentacion, respuesta_seleccionada,
                                        oracion['id'], comentario
                                    )
                                    if guardar_datos_retroalimentacion(st.session_state.datos_retroalimentacion):
                                        st.success("✅ Guardado individual")

                            with columna_btn2:
                                if oraciones_similares and st.button("🔄 Generalizar",
                                                                     key=f"generalizar_{oracion['id']}"):
                                    st.session_state.datos_retroalimentacion, contador_aplicadas = aplicar_retroalimentacion_generalizada(
                                        st.session_state.datos_retroalimentacion, oracion, comentario,
                                        oraciones_similares
                                    )
                                    if guardar_datos_retroalimentacion(st.session_state.datos_retroalimentacion):
                                        st.success(f"✅ Aplicado a {contador_aplicadas} oraciones similares")
                                        st.rerun()

                    st.write("")  # Espaciado

        else:
            st.warning("No se encontraron respuestas válidas")

    else:
        st.info("👆 Sube un archivo Excel para comenzar")

        # Ejemplo de formato
        st.subheader("📋 Formato del archivo Excel")
        st.write("""
        **El sistema funciona con cualquier archivo Excel de Google Forms:**
        - ✅ Solo necesita **una columna con las respuestas** (será la primera columna)
        - ✅ El **título de la columna** será mostrado como la pregunta
        - ✅ Cada fila representa una respuesta de un estudiante
        - ✅ El sistema automáticamente dividirá las respuestas largas en oraciones
        """)

        st.info(
            "💡 **Tip:** El nombre de la columna puede ser tan largo como se necesite (como las preguntas de Google Forms)")

        st.subheader("🤖 Funcionalidades de IA")
        st.write("""
        - 🔄 **Generalización manual**: Aplica retroalimentación a oraciones similares seleccionadas
        - 📊 **Análisis de similitud**: Detecta automáticamente contenido similar entre respuestas
        - 💾 **Persistencia**: Guarda y carga retroalimentación entre sesiones
        - ⚖️ **Umbral único**: Un solo parámetro controla toda la similitud del sistema
        """)

        st.subheader("📄 Archivos de ejemplo compatibles")
        st.write("""
        - 📊 **Google Forms** exportado como Excel
        - 📊 **Microsoft Forms** exportado como Excel  
        - 📊 **Cualquier Excel** con respuestas en la primera columna
        - 📊 **Encuestas** con preguntas abiertas
        """)


if __name__ == "__main__":
    principal()