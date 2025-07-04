import os
from fastapi import HTTPException
from google import genai
from ..handlers.Logger import logger
from ..utils.errors_feedback_builder import (
    FEEDBACK_JOINT_EXCEEDS_THRESHOLD,
    FEEDBACK_JOINT_WITHIN_LIMITS
)
from dotenv import load_dotenv
load_dotenv()

def transform_dict_vel_acel_to_text(dict_data: dict) -> str:
    """
    Convierte un diccionario en una cadena de texto.
    """
    text = ""
    for joint in dict_data:
        text += f"Articulación: {joint} {dict_data[joint]}\n"
               
    return text

def transform_dict_pos_ang_to_text(dict_data: dict) -> str:
    """
    Convierte un diccionario en una cadena de texto.
    """
    text = ""
    for joint in dict_data:
        text += f"Articulación: {joint}\n"
        for subdict in dict_data[joint]:
            text += f"inicio {subdict['s_inicio']}s fin {subdict['s_final']}s label {subdict['label']}\n"
    return text

async def get_feedback_from_external_IA(message: str) -> str:
    """
    Envía un mensaje a un servicio externo de IA y devuelve la respuesta.
    """
    # Aquí se implementaría la lógica para enviar el mensaje a un servicio externo de IA
    # y recibir la respuesta. Por ahora, solo se devuelve el mensaje original.
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_KEY"))

        response_not_processed = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=message,
        )

        final_response = response_not_processed.candidates[0].content.parts[0].text
        return final_response
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error al comunicarse con el servicio de IA externo.")


async def get_feedback_from_type_data_external_IA(message, type) -> str:
    """
    Envía datos a un servicio externo de IA y devuelve la respuesta.
    """
    # Aquí se implementaría la lógica para enviar los datos a un servicio externo de IA
    # y recibir la respuesta. Por ahora, solo se devuelve el mensaje original.
    # Promt de posicion: A continuación necesito un feedback de patrones biomecanicos (posiciones de articulaciones) donde el resultado lo vas a armar por cadenas de movimiento (como brazo, cuello, pierna, etc) te pasare las articulaciones con datos  donde  tendrás segundos iniciales, finales y la calificación de ventanas de tiempo del movimiento. ponlo en palabras sencillas y poco tecnicas; no hagas una introducción o una pregunta donde sugieras algún otro mensaje, esta respuesta ira conectada a una API y no debe notarse que una IA externa responde, no me des recomendaciones sobre enviarte otro mensaje de ningún tipo, tampoco un resumen general, si quieres sugerir mejoras que se enfoquen en como mejorar el movimiento del sujeto 2 en comparación del 1, pero hasta ahí, que la persona que lea no sienta que lo hizo un humano o robot, que solo sea el texto: , no debe ser muy largo
    # Promt de angular: "A continuación necesito un feedback de patrones biomecanicos (angulos de las articulaciones) donde el resultado lo vas a armar por cadenas de movimiento (como brazo, cuello, pierna, etc) te pasare las articulaciones con datos  donde  tendrás segundos iniciales, finales y la calificación de ventanas de tiempo del movimiento. ponlo en palabras sencillas y poco tecnicas; no hagas una introducción o una pregunta donde sugieras algún otro mensaje, esta respuesta ira conectada a una API y no debe notarse que una IA externa responde, no me des recomendaciones sobre enviarte otro mensaje de ningún tipo, tampoco un resumen general, si quieres sugerir mejoras que se enfoquen en como mejorar el movimiento del sujeto 2 en comparación del 1, pero hasta ahí, que la persona que lea no sienta que lo hizo un humano o robot, que solo sea el texto: , no debe ser muy largo"
    # Promt de acel vel: "A continuación necesito un feedback de patrones biomecanicos (aceleracion y velocidad) donde el resultado lo vas a armar por cadenas de movimiento (como brazo, cuello, pierna, etc) te pasare las articulaciones con datos  donde       VM: Velocidad media AM: Aceleración media MB: Movimiento brusco MS: Movimiento suave; que nacen de comparar el movimiento de un sujeto 2 a comparacion de otro, ponlo en palabras sencillas y poco tecnicas; no hagas una introducción o una pregunta donde sugieras algún otro mensaje, esta respuesta ira conectada a una API y no debe notarse que una IA externa responde, no me des recomendaciones sobre enviarte otro mensaje de ningún tipo, tampoco un resumen general, si quieres sugerir mejoras que se enfoquen en como mejorar el movimiento del sujeto 2 en comparación del 1, pero hasta ahí, que la persona que lea no sienta que lo hizo un humano o robot, que solo sea el texto: , no debe ser muy largo"
    try:
        instruction = ""
        if type == "position":
            instruction = """
                A continuación necesito un feedback de patrones biomecánicos enfocado hacia posiciones de articulaciones donde el resultado lo vas a armar por cadenas de movimiento funcional (como brazo, cuello, pierna, etc). Te pasaré las articulaciones con datos, donde tendrás segundos iniciales, finales y la calificación de ventanas de tiempo del movimiento.

                Ponlo en palabras sencillas y poco técnicas. No hagas una introducción ni una pregunta, no sugieras enviar ningún otro mensaje. Esta respuesta irá conectada a una API, por lo tanto no debe notarse que la escribe una IA externa ni un humano. No uses saludos ni cierres, solo el texto.

                Utiliza exactamente la siguiente estructura de secciones. Cada sección debe iniciar con su título en el formato "**[Título]:**", seguido por una o más frases descriptivas:

                **Cabeza y Cuello:**
                [Texto sobre articulaciones del cuello y cabeza]

                **Tronco:**
                [Texto sobre columna torácica, lumbar y sacroilíaca]

                **Brazo y Hombro Izquierdo:**
                [Texto sobre hombro izquierdo, codo izquierdo y muñeca izquierda]

                **Brazo y Hombro Derecho:**
                [Texto sobre hombro derecho, codo derecho y muñeca derecha]

                **Caderas y Piernas:**
                [Texto sobre caderas, rodillas y fémur]

                **Tobillos y Pies:**
                [Texto sobre tobillos, pies y estructuras relacionadas]

                **Recomendaciones Generales:**
                [Texto con sugerencias puntuales sobre cómo mejorar el movimiento del sujeto 2 comparado con el 1]

                Mantén esta estructura en todos los casos. No omitas ninguna sección, aunque el contenido de alguna sea breve. El análisis no debe ser muy largo, pero sí claro y especifico.
                """
        elif type == "angular":
            instruction = """
                A continuación necesito un feedback de patrones biomecanicos enfocado hacia angulos de las articulaciones donde el resultado lo vas a armar por cadenas de movimiento (como brazo, cuello, pierna, etc). Te pasaré las articulaciones con datos, donde tendrás segundos iniciales, finales y la calificación de ventanas de tiempo del movimiento.

                Ponlo en palabras sencillas y poco técnicas. No hagas una introducción ni una pregunta, no sugieras enviar ningún otro mensaje. Esta respuesta irá conectada a una API, por lo tanto no debe notarse que la escribe una IA externa ni un humano. No uses saludos ni cierres, solo el texto.

                Utiliza exactamente la siguiente estructura de secciones. Cada sección debe iniciar con su título en el formato "**[Título]:**", seguido por una o más frases descriptivas:

                **Cabeza y Cuello:**
                [Texto sobre articulaciones del cuello y cabeza]

                **Tronco:**
                [Texto sobre columna torácica, lumbar y sacroilíaca]

                **Brazo y Hombro Izquierdo:**
                [Texto sobre hombro izquierdo, codo izquierdo y muñeca izquierda]

                **Brazo y Hombro Derecho:**
                [Texto sobre hombro derecho, codo derecho y muñeca derecha]

                **Caderas y Piernas:**
                [Texto sobre caderas, rodillas y fémur]

                **Tobillos y Pies:**
                [Texto sobre tobillos, pies y estructuras relacionadas]

                **Recomendaciones Generales:**
                [Texto con sugerencias puntuales sobre cómo mejorar el movimiento del sujeto 2 comparado con el 1 en cuestion de angulos de las articulaciones]

                Mantén esta estructura en todos los casos. No omitas ninguna sección, aunque el contenido de alguna sea breve. El análisis no debe ser muy largo, pero sí claro y especifico.
                """
        elif type == "acel_vel":
            instruction = """
                A continuación necesito un feedback de patrones biomecanicos enfocado hacia aceleracion y velocidad del movimiento donde el resultado lo vas a armar por cadenas de movimiento (como brazo, cuello, pierna, etc). Te pasaré las articulaciones con datosA continuación necesito un feedback de patrones biomecanicos (aceleracion y velocidad) donde el resultado lo vas a armar por cadenas de movimiento (como brazo, cuello, pierna, etc) te pasare las articulaciones con datos  donde, VM: Velocidad media AM: Aceleración media MB: Movimiento brusco MS: Movimiento suave; que nacen de comparar el movimiento de un sujeto 2 a comparacion de otro

                Ponlo en palabras sencillas y poco técnicas. No hagas una introducción ni una pregunta, no sugieras enviar ningún otro mensaje. Esta respuesta irá conectada a una API, por lo tanto no debe notarse que la escribe una IA externa ni un humano. No uses saludos ni cierres, solo el texto.

                Utiliza exactamente la siguiente estructura de secciones. Cada sección debe iniciar con su título en el formato "**[Título]:**", seguido por una o más frases descriptivas:

                **Cabeza y Cuello:**
                [Texto sobre articulaciones del cuello y cabeza]

                **Tronco:**
                [Texto sobre columna torácica, lumbar y sacroilíaca]

                **Brazo y Hombro Izquierdo:**
                [Texto sobre hombro izquierdo, codo izquierdo y muñeca izquierda]

                **Brazo y Hombro Derecho:**
                [Texto sobre hombro derecho, codo derecho y muñeca derecha]

                **Caderas y Piernas:**
                [Texto sobre caderas, rodillas y fémur]

                **Tobillos y Pies:**
                [Texto sobre tobillos, pies y estructuras relacionadas]

                **Recomendaciones Generales:**
                [Texto con sugerencias puntuales sobre cómo mejorar el movimiento del sujeto 2 comparado con el 1 en cuestion de angulos de las articulaciones]

                Mantén esta estructura en todos los casos. No omitas ninguna sección, aunque el contenido de alguna sea breve. El análisis no debe ser muy largo, pero sí claro y especifico.
                """
        else:
            raise RuntimeError("Tipo de datos no soportado.")
        
        return await get_feedback_from_external_IA(f"{instruction} {message}")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"{e} Error al generar mensaje para el servicio de IA externo.")


"""
Genera mensaje de retroalimentación basado en errores y umbrales.
"""
def build_feedback(joint_errors: dict, clusters: dict, thresholds: dict):
    feedback = {}
    for joint, errors in joint_errors.items():
        rmse = errors.get('rmse')
        if rmse > thresholds.get(joint, float('inf')):
            feedback[joint] = FEEDBACK_JOINT_EXCEEDS_THRESHOLD.format(joint, rmse)
        else:
            feedback[joint] = FEEDBACK_JOINT_WITHIN_LIMITS.format(joint)
    return feedback