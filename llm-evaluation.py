import torch
import pandas as pd
import os
import openai
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import evaluate
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PyPDF2 import PdfReader

# 🔹 1. Charger les questions/réponses depuis le fichier Excel
qa_file = "QR/Q&A.xlsx"
df = pd.read_excel(qa_file)

# Vérifier la structure
if "Question" not in df.columns or "Réponse Attendue" not in df.columns:
    raise ValueError("Le fichier Excel doit contenir les colonnes 'Question' et 'Réponse Attendue'.")

questions = df["Question"].tolist()
expected_answers = df["Réponse Attendue"].tolist()

# 🔹 2. Charger le modèle T5 depuis Hugging Face pour la génération de réponse attendue
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

# 🔹 3. Initialiser la clé API MistralAI
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
openai.api_key = MISTRAL_API_KEY

# 🔹 4. Fonction pour générer une réponse à partir du modèle MistralAI via l'API
def generate_response_mistral_api(prompt):
    response = openai.Completion.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# 🔹 5. Fonction pour générer la réponse attendue avec T5
def generate_expected_answer_with_t5(question):
    input_text = "generate answer: " + question
    inputs = t5_tokenizer(input_text, return_tensors="pt")
    output = t5_model.generate(**inputs, max_length=50)
    return t5_tokenizer.decode(output[0], skip_special_tokens=True)

# 🔹 6. Fonction pour extraire du texte d'un fichier PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# 🔹 7. Fonction pour calculer la cohérence contextuelle entre la réponse générée et le contexte des documents
def compute_contextual_coherence(generated, expected, context):
    # Utilisation d'un modèle pré-entrainé pour les embeddings
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    generated_embedding = model.encode(generated, convert_to_tensor=True)
    expected_embedding = model.encode(expected, convert_to_tensor=True)
    context_embedding = model.encode(context, convert_to_tensor=True)
    
    # Calcul de la similarité cosinus
    similarity_generated_expected = cosine_similarity([generated_embedding], [expected_embedding])
    similarity_generated_context = cosine_similarity([generated_embedding], [context_embedding])
    
    # Retourner une mesure combinée de la cohérence
    coherence_score = np.mean([similarity_generated_expected[0][0], similarity_generated_context[0][0]])
    return coherence_score

# 🔹 8. Initialiser les métriques
bleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")
perplexity_model = evaluate.load("perplexity", module_type="metric")

# 🔹 9. Chargement des documents et extraction de texte
documents_folder = "data/documents"
documents_text = []
for filename in os.listdir(documents_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(documents_folder, filename)
        text = extract_text_from_pdf(pdf_path)
        documents_text.append(text)

# 🔹 10. Évaluation du modèle pour chaque question
generated_answers = []
bleu_scores = []
meteor_scores = []
bertscore_scores = []
perplexity_scores = []
coherence_scores = []

for i, question in enumerate(questions):
    print(f"📝 Question {i+1}/{len(questions)}: {question}")

    # Générer la réponse du modèle MistralAI via l'API
    generated_mistral = generate_response_mistral_api(question)
    generated_answers.append(generated_mistral)
    
    # Générer la réponse attendue avec T5
    expected_t5 = generate_expected_answer_with_t5(question)
    
    # Calculer la cohérence contextuelle en utilisant le texte extrait des documents PDF
    context = " ".join(documents_text[:5])  # On peut ajuster le nombre de documents utilisés ici
    coherence_score = compute_contextual_coherence(generated_mistral, expected_t5, context)
    coherence_scores.append(coherence_score)

    # Évaluation avec BLEU
    bleu_result = bleu.compute(predictions=[generated_mistral], references=[[expected_answers[i]]])
    bleu_scores.append(bleu_result["score"])

    # METEOR Score
    meteor_result = meteor.compute(predictions=[generated_mistral], references=[expected_answers[i]])
    meteor_scores.append(meteor_result["meteor"])

    # BERTScore
    bertscore_result = bertscore.compute(predictions=[generated_mistral], references=[expected_answers[i]], lang="fr")
    bertscore_scores.append(bertscore_result["f1"][0])

    # Perplexité
    tokens = t5_tokenizer(generated_mistral, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = t5_model(tokens).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    perplexity = math.exp(-log_probs.mean().item())
    perplexity_scores.append(perplexity)

# 🔹 11. Sauvegarder les résultats dans un fichier Excel
df_results = pd.DataFrame({
    "Question": questions,
    "Réponse Attendue": expected_answers,
    "Réponse Mistral": generated_answers,
    "Score BLEU": bleu_scores,
    "Score METEOR": meteor_scores,
    "Score BERTScore": bertscore_scores,
    "Perplexité": perplexity_scores,
    "Cohérence Contextuelle": coherence_scores
})

# Sauvegarde des résultats
output_file = "Evaluation_Mistral_with_T5_and_Contextual_Cohesion.xlsx"
df_results.to_excel(output_file, index=False)

print(f"📊 Évaluation terminée. Résultats enregistrés dans {output_file}.")
