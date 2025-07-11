import pandas as pd
import random
import gradio as gr
import spacy
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Cargar datos
df = pd.read_csv("dataEducacion_final_393_agregado_2.csv")

# Cargar modelo entrenado
model = joblib.load("modelo_entrenado.pkl")
 
# Cargar SpaCy en español
nlp = spacy.load("es_core_news_sm")
 
# Normalizador de texto
def normalize_text(text):
    text = text.lower()
    text = unidecode.unidecode(text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Diccionarios de saludo y despedida
saludos = ["hola", "buenos dias", "buenas tardes", "buenas noches", "hola que tal", "como estas", "ey"]
respuestas_saludo = ["¡Hola! ¿En qué puedo ayudarte?", "¡Buenos días! ¿Qué necesitas?", "Hola, dime tu duda."]
despedidas = ["gracias", "hasta luego", "adios", "nos vemos", "bye", "chao"]
respuestas_despedida = ["¡Hasta pronto!", "Gracias por tu consulta. ¡Éxitos!", "Nos vemos."]
 
# Función principal de respuesta
def responder(pregunta):
    pregunta_limpia = normalize_text(pregunta)
    if any(s in pregunta.lower() for s in saludos):
        return random.choice(respuestas_saludo)

    if any(d in pregunta.lower() for d in despedidas):
        return random.choice(respuestas_despedida)
 
    etiqueta = model.predict([pregunta_limpia])[0]
    respuestas_categoria = df[df["categoria"] == etiqueta].copy()
    respuestas_categoria["respuesta_limpia"] = respuestas_categoria["respuestas_compuestas"].apply(normalize_text)
 
    vectorizer = model.named_steps["tfidfvectorizer"]
    respuestas_categoria["respuesta_vec"] = respuestas_categoria["respuesta_limpia"].apply(lambda x: vectorizer.transform([x]))
    pregunta_vec = vectorizer.transform([pregunta_limpia])
    respuestas_categoria["similitud"] = respuestas_categoria["respuesta_vec"].apply(lambda x: cosine_similarity(x, pregunta_vec)[0][0])
    mejor_idx = respuestas_categoria["similitud"].idxmax()
    return df.loc[mejor_idx, "respuesta"]
 



# Interfaz con Gradio
iface = gr.Interface(
    fn=responder, inputs=gr.Textbox(label="Tu pregunta"), 
    outputs=gr.Textbox(label="Respuesta del chatbot"),
    title="Chat Educativo Interactivo",
    description="Haz preguntas sobre matrículas, clases, pagos y más. Respuestas automatizadas con IA educativa.")

iface.launch()