import os
import re 
import json
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# 모델 및 사전 준비
# GPT
client = OpenAI(api_key="sk-proj-b2x05FmtNy1Q2cfocAUGT3BlbkFJvmLMXpUHJM5TXCyb7o95")
# BERT
model_directory = './emotion_prediction_bert'
tokenizer = BertTokenizer.from_pretrained(model_directory)
model = BertForSequenceClassification.from_pretrained(model_directory)
id_to_label = {
    0: "두려워하는", 1: "화난", 2: "당황스러운", 3: "실망한", 4: "감상적인",
    5: "놀란", 6: "신뢰하는", 7: "흥분한", 8: "자랑스러운", 9: "슬픈",
    10: "외로운", 11: "고마워하는", 12: "기대하는", 13: "감명받은",
    14: "즐거운", 15: "희망에 찬", 16: "역겨운", 17: "확신하는", 18: "질투하는",
    19: "준비된", 20: "만족하는", 21: "돌보는"
}
negative_emotions = {"두려워하는", "화난", "당황스러운", "실망한", "놀란", "슬픈", "외로운", "역겨운"}
middle_emotions = {"감상적인", "흥분한", "돌보는", "질투하는"}
positive_emotions = {"신뢰하는", "자랑스러운", "고마워하는", "기대하는", "감명받은", "즐거운", "희망에 찬", "확신하는", "준비된", "만족하는"}

def generate_answers(occupation, question, model="gpt-4o", max_tokens=300, num_answers=5):
    responses = []
    
    query = f"""
    - job information: {occupation},
    - interview question: {question}
    """
    
    for _ in range(num_answers):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an assistant that helps with interview preparation. Based on the given interview question and job information, assist the interviewee in crafting good responses."},
                {"role": "user", "content": query}
            ],
            max_tokens=max_tokens
        )
        answer = response.choices[0].message.content.strip()
        responses.append(answer)
    return responses

def calculate_scores(text1, text2):
    vectorizer = TfidfVectorizer()
    all_texts = [text1, text2]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    svd = TruncatedSVD(n_components=100)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    
    text1_lsa = lsa_matrix[0:1]
    text2_lsa = lsa_matrix[1:]
    
    lsa_scores = np.sum(text2_lsa, axis=1) / text2_lsa.shape[1]
    average_lsa_score = np.mean(lsa_scores)
    
    cosine_similarities = cosine_similarity(text1_lsa, text2_lsa)
    average_cosine_score = np.mean(cosine_similarities)
    
    return average_lsa_score, average_cosine_score

def scale_scores_coqna(lsa_score, cossim_score, max_score=25):
    lsa_scale_factor = 300 
    cossim_scale_factor = 100 
    
    scaled_lsa_score = min(lsa_score * lsa_scale_factor, max_score)
    scaled_cossim_score = min(cossim_score * cossim_scale_factor, max_score)
    
    return scaled_lsa_score, scaled_cossim_score

def scale_scores_cover_letter(lsa_score, cossim_score, max_score=25):
    lsa_scale_factor = 1000  
    cossim_scale_factor = 300
    
    scaled_lsa_score = min(lsa_score * lsa_scale_factor, max_score)
    scaled_cossim_score = min(cossim_score * cossim_scale_factor, max_score)
    
    return scaled_lsa_score, scaled_cossim_score

# 문맥 분석 함수
def evaluation(question, answer, model = "gpt-4o"):
    query= f"""
        Write an assessment report for the interview Question and Answer. Please provide your response:
        {{
            "Interview Question": Question, 
            "Assessment items": {{ "Score"\n, "Details" }}
        }}
        
        Instructions:
        - Format: Json
        - Level of Difficulty: Advanced
        - Target Audience: Applicant
        - Assessment items: Relevance, Logicality, Clarity, Question Comprehension
        - Show the score and evaluation details in detail with each evaluation item/100 points.
        - Language Used: Detected and set by "Question"
        
        Resume Information
        - Question: {question}
        - Answer: {answer}
        
        Please respond in Language is detected and set by "Question".
        
    """

    messages = [
            {"role": "system", "content": "You are a helpful human resources assistant."},
            {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    evaluation = response.choices[0].message.content
    evaluation = re.sub(r'```json|```', '', evaluation).strip()
    print(evaluation)
    print("="*20)
    
    # ===== 후처리 추가 =====
    evaluation = json.loads(evaluation)
    print(evaluation)
    
    munmek_score = 0
    result = dict()
    
    columns = ['Relevance', 'Logicality', 'Clarity', 'Question Comprehension']
    for column in columns:
        munmek_score += evaluation['Assessment items'][column]['Score']
        result[column] = evaluation['Assessment items'][column]['Details']
        
    # score 계산
    munmek_score /= len(columns)
    munmek_score = munmek_score / 100 * 40
    result['munmek_score'] = munmek_score
    
    return result

# 감정 분석
def calculate_score(emotion, confidence):
    if emotion in negative_emotions:
        score = 10 * (1 - confidence)
    elif emotion in middle_emotions:
        score = 5 + 5 * confidence
    elif emotion in positive_emotions:
        score = 10 * confidence
    else:
        score = 0  
    return score
def predict_emotion_and_score(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    predicted_emotion = id_to_label[predicted_class_id]
    confidence_score = probabilities[0][predicted_class_id].item()
    
    emotion_score = calculate_score(predicted_emotion, confidence_score)
    
    return predicted_emotion, confidence_score, emotion_score

# 최종 result 계산
def calculate_result(scaled_lsa_score, scaled_cossim_score, munmek, emotion_score):
    result = {
        "lsa_score": scaled_lsa_score,
        "similarity_score": scaled_cossim_score,
        "munmek_score" : munmek['munmek_score'],
        "munmek" : munmek,
        "emotion_score": emotion_score,
        "context_score": scaled_lsa_score + scaled_cossim_score + munmek['munmek_score'] + emotion_score
    }
    
    return result

# ====== 라우터 ======
@app.route('/coQnaEval', methods=['POST'])
def co_qna_eval():
    print("시작", "="*20)
    data = request.json
    occupation = data['occupation']
    question = data['question']
    answer = data['answer']
    
    # Cosine Similarity, lsa 분석
    generated_answers = generate_answers(occupation, question)
    lsa_score, cossim_score = calculate_scores(answer, generated_answers[0])
    scaled_lsa_score, scaled_cossim_score = scale_scores_coqna(lsa_score, cossim_score)
    
    # 문맥 분석
    munmek = evaluation(question, answer)
    
    # 감정 분석
    _, _, emotion_score = predict_emotion_and_score(answer)
    
    result = calculate_result(scaled_lsa_score, scaled_cossim_score, munmek, emotion_score)
    
    print(result)
    
    return jsonify(result)

@app.route('/coverLetterEval', methods=['POST'])
def cover_letter_eval():
    print("시작", "="*20)
    data = request.json
    question = data['question']
    answer = data['answer']
    cover_letter = data['cover_letter']
    
    # Cosine Similarity, lsa 분석
    lsa_score, cossim_score = calculate_scores(answer, cover_letter)
    scaled_lsa_score, scaled_cossim_score = scale_scores_cover_letter(lsa_score, cossim_score)
    
    # 문맥 분석
    munmek = evaluation(question, answer)
    
    # 감정 분석
    _, _, emotion_score = predict_emotion_and_score(answer)
    
    result = calculate_result(scaled_lsa_score, scaled_cossim_score, munmek, emotion_score)
    
    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
