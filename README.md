# FAQ Chatbot Using NLP

An intelligent FAQ Chatbot developed using Python, Tkinter, NLTK, and Scikit-learn. The chatbot analyzes user queries with Natural Language Processing (NLP) techniques and provides the most relevant answer by matching the query with a predefined FAQ dataset using TF-IDF Vectorization and Cosine Similarity.

## Features

- NLP-based text preprocessing
- Tokenization and lemmatization using NLTK
- Stop-word removal and text cleaning
- TF-IDF vectorization
- Cosine similarity for FAQ matching
- Interactive chat interface built with Tkinter
- Automatic suggestions for unmatched queries
- User-friendly and responsive design

## Technologies Used

- Python
- Tkinter
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

## Project Structure

```
CodeAlpha_ChatbotforFAQs/
│
├── chatbot.py
├── show_output.py
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/rimshyybutt/CodeAlpha_ChatbotforFAQs.git
```

Navigate to the project folder:

```bash
cd CodeAlpha_ChatbotforFAQs
```

Install the required dependencies:

```bash
pip install nltk scikit-learn
```

Run the chatbot:

```bash
python chatbot.py
```

## Sample Questions

- What products do you sell?
- Do you offer international shipping?
- How long does delivery take?
- Can I return a product?
- How can I contact support?
- Do you offer customization?

## How It Works

1. The user enters a question.
2. The text is preprocessed using NLP techniques.
3. TF-IDF converts the text into numerical vectors.
4. Cosine similarity identifies the closest matching FAQ.
5. The chatbot displays the corresponding answer.

## Project Objective

This project demonstrates the application of Natural Language Processing techniques to build an intelligent FAQ chatbot that provides quick and accurate responses to user queries through a simple graphical interface.

## Author

Rimsha Butt

Developed as part of the CodeAlpha Internship Program.
