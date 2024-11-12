from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json

app = Flask(__name__)

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your database
with open('questions_answers.json', 'r') as file:
    data = json.load(file)

# Precompute embeddings for correct answers
for question in data['questions']:
    question['answer_embeddings'] = model.encode(question['answers'], convert_to_tensor=True)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    content = request.json
    question_id = content.get('question_id')
    user_response = content.get('user_response')
    
    if not question_id or not user_response:
        return jsonify({"error": "Missing question_id or user_response"}), 400
    
    # Find the question
    question = next((q for q in data['questions'] if q['id'] == question_id), None)
    if not question:
        return jsonify({"error": "Question not found"}), 404
    
    # Encode user response
    user_embedding = model.encode(user_response, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_scores = util.pytorch_cos_sim(user_embedding, question['answer_embeddings'])
    max_score = cosine_scores.max().item()
    
    # Determine adequacy
    threshold = 0.75
    is_adequate = max_score >= threshold
    
    # Generate feedback
    if is_adequate:
        feedback = "Good response! Your answer aligns well with the ideal responses."
    else:
        feedback = "Your response could be improved. Try to elaborate more on key points."
    
    return jsonify({
        "similarity_score": round(max_score, 2),
        "is_adequate": is_adequate,
        "feedback": feedback
    })
#
if __name__ == '__main__':
    app.run(debug=True)


