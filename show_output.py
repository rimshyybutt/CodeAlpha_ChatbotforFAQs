import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure NLTK resources
for resource in ['punkt', 'wordnet', 'omw-1.4', 'stopwords']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

faqs = {
    "What products do you sell?": "We sell candles, candle bouquets, and handmade decorative items.",
    "Do you offer international shipping?": "Yes, we ship our products worldwide.",
    "How long does delivery take?": "Delivery usually takes 3–7 working days depending on the destination.",
    "What payment methods do you accept?": "We accept bank transfer, credit/debit cards, and cash on delivery.",
    "Can I return a product?": "Yes, returns are accepted within 7 days if the product is unused and in original condition.",
    "Do you offer customization?": "Yes, we offer full customization for candle designs, gift boxes, and packaging.",
    "How can I contact support?": "You can contact us via email, WhatsApp, or our support form.",
    "Do you provide bulk order discounts?": "Yes, bulk orders qualify for special pricing and priority fulfillment.",
    "Can I change my order after purchase?": "If your order has not shipped yet, we can usually update it. Please contact support immediately.",
    "Are your products eco-friendly?": "Yes, we use sustainable materials and recyclable packaging whenever possible."
}

questions = list(faqs.keys())
answers = list(faqs.values())

# Preprocess (same as chatbot)
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

clean_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(clean_questions)

def get_response(user_input):
    if not user_input.strip():
        return '🤖 Please ask a question to get started.'
    user_input_clean = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(user_vec, X)[0]
    best_index = similarity.argmax()
    score = similarity[best_index]
    if score < 0.20:
        suggestions = '\n'.join([f'- {q}' for q in questions[:4]])
        return ('🤖 I couldn\'t match your question with high confidence. Try one of these common questions or rephrase your request:\n' + suggestions)
    return answers[best_index]

# Sample interactions
samples = [
    "Hi, can you help me?",
    "Do you offer international shipping?",
    "I want to change my order, is that possible?",
    "What payment methods do you accept?",
    "Are your products eco friendly?",
    "Do you sell electronics?",
    "Can I return a product if it is unused?",
    "What products do you sell?",
    "Do you provide bulk order discounts?",
    "How can I contact support?",
    "Can I customize a candle for a gift?",
    "Do you have any gift wrapping options?",
    "How much does shipping cost?",
    "Can I cancel my order before it ships?",
    "What are your working hours?",
    "Do you ship to Canada?",
    "Is it possible to order in bulk?",
    "Do you make scented candles?",
    "Can I get a refund if I change my mind?",
    "What is your return policy?",
    "How long does it take to make an order?",
    "Do you offer same day delivery?",
    "Can I contact you on WhatsApp?",
    "Are your materials recyclable?",
    "Is the packaging eco friendly?"
]

print('\n--- Sample conversation outputs ---\n')
for s in samples:
    print('User:', s)
    print('Bot :', get_response(s))
    print()
