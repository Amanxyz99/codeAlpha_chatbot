# FAQ Chatbot using NLP and Cosine Similarity

import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# FAQ data
faqs = {
    "What is AI?": "AI stands for Artificial Intelligence.",
    "What is Machine Learning?": "Machine Learning is a subset of AI.",
    "What is NLP?": "NLP stands for Natural Language Processing.",
    "What is Python?": "Python is a popular programming language used in AI and web development.",
    "What is a chatbot?": "A chatbot is a software that simulates human conversation."
}

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Prepare questions
questions = list(faqs.keys())
processed_questions = [preprocess(q) for q in questions]

# Vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

# Function to get chatbot response
def get_answer(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarity.argmax()
    return list(faqs.values())[best_match_index]

# Chatbot loop
print("🤖 FAQ Chatbot (type 'exit' to quit)")

while True:
    user_question = input("You: ")
    if user_question.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break
    response = get_answer(user_question)
    print("Bot:", response)
