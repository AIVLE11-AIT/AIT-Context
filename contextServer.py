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
        try:
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
        except Exception as e:
            print(f"Error in generating answer: {e}")
            responses.append("")

    print("Generated Answers:", responses)
    return responses

def calculate_scores(text1, text2):
    if not text1 or not text2 or text1.strip() == "" or text2.strip() == "":
        return 0, 0

    vectorizer = TfidfVectorizer()
    all_texts = [text1, text2]
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    n_features = tfidf_matrix.shape[1]
    n_components = min(100, n_features)

    if n_components < 1:
        n_components = 1

    svd = TruncatedSVD(n_components=n_components)
    lsa_matrix = svd.fit_transform(tfidf_matrix)

    text1_lsa = lsa_matrix[0:1]
    text2_lsa = lsa_matrix[1:]

    lsa_scores = np.sum(text2_lsa, axis=1) / text2_lsa.shape[1]
    average_lsa_score = np.mean(lsa_scores)

    cosine_similarities = cosine_similarity(text1_lsa, text2_lsa)
    average_cosine_score = np.mean(cosine_similarities)

    average_lsa_score = max(average_lsa_score, 0)
    average_cosine_score = max(average_cosine_score, 0)

    print("LSA Score:", average_lsa_score)
    print("Cosine Similarity Score:", average_cosine_score)

    return average_lsa_score, average_cosine_score

def normalize_score(score, max_possible_value):
    return score / max_possible_value

def scale_scores_coqna(lsa_score, cossim_score, max_score=20):
    normalized_lsa_score = normalize_score(lsa_score, 1)
    normalized_cossim_score = normalize_score(cossim_score, 1)

    # LSA score를 1.8배로 증가
    scaled_lsa_score = min(normalized_lsa_score * 10 * 3.5, max_score)
    # Cosine similarity score를 5배로 증가
    scaled_cossim_score = min(normalized_cossim_score * 10 * 10, max_score)

    return scaled_lsa_score, scaled_cossim_score

def scale_scores_cover_letter(lsa_score, cossim_score, max_score=20):
    normalized_lsa_score = normalize_score(lsa_score, 1)
    normalized_cossim_score = normalize_score(cossim_score, 1)

    # LSA score를 1.8배로 증가
    scaled_lsa_score = min(normalized_lsa_score * 10 * 3.5, max_score)
    # Cosine similarity score를 5배로 증가
    scaled_cossim_score = min(normalized_cossim_score * 10 * 10, max_score)

    return scaled_lsa_score, scaled_cossim_score

def evaluation(question, answer, model="gpt-4o"):
    query = f"""
        Write an assessment report for the interview Question and Answer. Please provide your response:
        {{
            "Interview Question": "{question}",
            "Assessment Items": {{
                "Relevance": {{"Score": null, "Details": ""}},
                "Logicality": {{"Score": null, "Details": ""}},
                "Clarity": {{"Score": null, "Details": ""}},
                "Question Comprehension": {{"Score": null, "Details": ""}}
            }}
        }}

        Instructions:
        - Format: Json
        - Level of Difficulty: Advanced
        - Target Audience: Applicant
        - Assessment Items: Relevance, Logicality, Clarity, Question Comprehension
        - Show the score and evaluation details in detail with each evaluation item/100 points.
        - Language Used: Detected and set by "Question".

        Resume Information
        - Question: {question}
        - Answer: {answer}

        Please respond in the language detected and set by "Question".
    """

    messages = [
        {"role": "system", "content": "You are a helpful human resources assistant."},
        {"role": "user", "content": query}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        evaluation = response.choices[0].message.content
        print(f"Raw JSON response: {evaluation}")
        
        # 정리 및 파싱
        evaluation = re.sub(r'```json|```', '', evaluation).strip()
        evaluation = evaluation.replace("\n", " ").replace("\r", "")
        evaluation = re.sub(r'\s+', ' ', evaluation)
        print(f"Stripped JSON response: {evaluation}")
        
        evaluation = json.loads(evaluation)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        print(f"Response content: {response.choices[0].message.content}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    munmek_score = 0
    result = dict()

    columns = ['Relevance', 'Logicality', 'Clarity', 'Question Comprehension']
    for column in columns:
        if column in evaluation.get('Assessment Items', {}):
            munmek_score += evaluation['Assessment Items'][column]['Score']
            result[column] = evaluation['Assessment Items'][column]['Details']
        else:
            result[column] = "No details provided"

    munmek_score /= len(columns)
    munmek_score = munmek_score / 100 * 40
    result['munmek_score'] = munmek_score

    print("Evaluation Result:", result)

    return result

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

    emotion_score = calculate_score(predicted_emotion, confidence_score) * 2

    print("Predicted Emotion:", predicted_emotion)
    print("Confidence Score:", confidence_score)
    print("Emotion Score:", emotion_score)

    return predicted_emotion, confidence_score, emotion_score

def calculate_result(scaled_lsa_score, scaled_cossim_score, munmek, emotion_score):
    context_score = max(scaled_lsa_score + scaled_cossim_score + munmek['munmek_score'] + emotion_score, 0)

    result = {
        "lsa_score": scaled_lsa_score,
        "similarity_score": scaled_cossim_score,
        "munmek_score": munmek['munmek_score'],
        "munmek": munmek,
        "emotion_score": emotion_score,
        "context_score": context_score
    }

    print("Final Result:", result)

    return result

def empty_answer_response(language):
    if language == "en":
        return {
            "lsa_score": 0,
            "similarity_score": 0,
            "munmek_score": 0,
            "munmek": {
                "Clarity": "No assessment provided.",
                "Logicality": "No assessment provided.",
                "Question Comprehension": "No assessment provided.",
                "Relevance": "No assessment provided."
            },
            "emotion_score": 0,
            "context_score": 0
        }
    else:
        return {
            "lsa_score": 0,
            "similarity_score": 0,
            "munmek_score": 0,
            "munmek": {
                "Clarity": "평가 항목 없음.",
                "Logicality": "평가 항목 없음.",
                "Question Comprehension": "평가 항목 없음.",
                "Relevance": "평가 항목 없음."
            },
            "emotion_score": 0,
            "context_score": 0
        }

@app.route('/coQnaEval', methods=['POST'])
def co_qna_eval():
    print("coQnaEval 시작", "="*20)
    data = request.get_json()

    if not data or 'occupation' not in data or 'question' not in data or 'answer' not in data:
        return jsonify({'context_score': 0}), 200

    occupation = data['occupation']
    question = data['question']
    answer = data['answer']

    print(f"Received answer: {answer}")

    if answer is None or not answer.strip() or answer.lower() in {"none", "null", "n/a"}:
        language = "en" if re.search("[a-zA-Z]", question) else "ko"
        print(f"Empty or invalid answer detected. Returning default response for language: {language}")
        return jsonify(empty_answer_response(language)), 200

    try:
        generated_answers = generate_answers(occupation, question)
        lsa_score, cossim_score = calculate_scores(answer, generated_answers[0])
        print(f"LSA Score: {lsa_score}, Cosine Similarity Score: {cossim_score}")
        scaled_lsa_score, scaled_cossim_score = scale_scores_coqna(lsa_score, cossim_score)
        print(f"Scaled LSA Score: {scaled_lsa_score}, Scaled Cosine Similarity Score: {scaled_cossim_score}")
    except ValueError as e:
        print(f"Error in calculate_scores or scale_scores_coqna: {e}")
        return jsonify({'context_score': 0}), 200

    try:
        munmek = evaluation(question, answer)
        if munmek is None:
            raise ValueError('Failed to evaluate the answer')
        print(f"Context Analysis Result: {munmek}")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return jsonify({'context_score': 0}), 200

    try:
        _, _, emotion_score = predict_emotion_and_score(answer)
        print(f"Emotion Score: {emotion_score}")
    except Exception as e:
        print(f"Error in predict_emotion_and_score: {e}")
        return jsonify({'context_score': 0}), 200

    try:
        result = calculate_result(scaled_lsa_score, scaled_cossim_score, munmek, emotion_score)
        print("coQnaEval Result:", result)
        return jsonify(result)
    except Exception as e:
        print(f"Error in calculate_result: {e}")
        return jsonify({'context_score': 0}), 200

@app.route('/coverLetterEval', methods=['POST'])
def cover_letter_eval():
    print("coverLetterEval 시작", "="*20)
    data = request.get_json()

    if not data or 'question' not in data or 'answer' not in data or 'cover_letter' not in data:
        return jsonify({'context_score': 0}), 200

    question = data['question']
    answer = data['answer']
    cover_letter = data['cover_letter']

    print(f"Received answer: {answer}")

    if answer is None or not answer.strip() or answer.lower() in {"none", "null", "n/a"}:
        language = "en" if re.search("[a-zA-Z]", question) else "ko"
        print(f"Empty or invalid answer detected. Returning default response for language: {language}")
        return jsonify(empty_answer_response(language)), 200

    try:
        lsa_score, cossim_score = calculate_scores(answer, cover_letter)
        print(f"LSA Score: {lsa_score}, Cosine Similarity Score: {cossim_score}")
        scaled_lsa_score, scaled_cossim_score = scale_scores_cover_letter(lsa_score, cossim_score)
        print(f"Scaled LSA Score: {scaled_lsa_score}, Scaled Cosine Similarity Score: {scaled_cossim_score}")
    except ValueError as e:
        print(f"Error in calculate_scores or scale_scores_cover_letter: {e}")
        return jsonify({'context_score': 0}), 200

    try:
        munmek = evaluation(question, answer)
        if munmek is None:
            raise ValueError('Failed to evaluate the answer')
        print(f"Context Analysis Result: {munmek}")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return jsonify({'context_score': 0}), 200

    try:
        _, _, emotion_score = predict_emotion_and_score(answer)
        print(f"Emotion Score: {emotion_score}")
    except Exception as e:
        print(f"Error in predict_emotion_and_score: {e}")
        return jsonify({'context_score': 0}), 200

    try:
        result = calculate_result(scaled_lsa_score, scaled_cossim_score, munmek, emotion_score)
        print("coverLetterEval Result:", result)
        return jsonify(result)
    except Exception as e:
        print(f"Error in calculate_result: {e}")
        return jsonify({'context_score': 0}), 200

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting the server: {e}")
