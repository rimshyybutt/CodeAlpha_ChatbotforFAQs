# ---------------- IMPORTS ----------------
import tkinter as tk
from tkinter import ttk
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- NLTK SETUP ----------------
for resource in ['punkt', 'wordnet', 'omw-1.4', 'stopwords']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ---------------- FAQ DATASET ----------------
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
    "Are your products eco-friendly?": "Yes, we use sustainable materials and recyclable packaging whenever possible.",
    "How do I track my order?": "You can track your order using the tracking link sent to your email after shipping.",
    "What should I do if my package is delayed?": "If your package is delayed, contact support with your order number for an update.",
    "Do you offer same-day delivery?": "Same-day delivery is available in select cities. Please contact us to confirm availability.",
    "Can I get a gift receipt?": "Yes, gift receipts are available upon request at checkout.",
    "What items are in stock?": "Stock availability is displayed on the product page, and popular items are restocked frequently.",
    "Is your packaging eco-friendly?": "Yes, we use recyclable materials and eco-friendly packaging whenever possible.",
    "Can I request a custom message with my order?": "Absolutely, custom gift messages are available during checkout.",
    "How do I cancel my order?": "Orders may be cancelled before shipping by contacting support immediately.",
    "Do you offer wholesale pricing?": "Yes, wholesale and bulk order pricing is available. Contact our sales team for details."
}

questions = list(faqs.keys())
answers = list(faqs.values())

# ---------------- PREPROCESSING ----------------
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

clean_questions = [preprocess(q) for q in questions]

# ---------------- TF-IDF MODEL ----------------
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(clean_questions)

# ---------------- CHATBOT ENGINE ----------------
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
        return (
            '🤖 I couldn\'t match your question with high confidence. '
            'Try one of these common questions or rephrase your request:\n' + suggestions
        )

    return answers[best_index]

# ---------------- SEND FUNCTION ----------------
def send_message(event=None):
    user_msg = entry.get().strip()
    if user_msg == '':
        return

    add_message(user_msg, sender='user')
    entry.delete(0, tk.END)
    send_btn.state(['disabled'])
    typing_label.config(text='Typing...')
    root.update_idletasks()
    root.after(120, lambda: finish_response(user_msg))


def finish_response(user_msg):
    bot_reply = get_response(user_msg)
    add_message(bot_reply, sender='bot')
    typing_label.config(text='')
    send_btn.state(['!disabled'])


def add_message(text, sender='bot'):
    bubble = tk.Frame(messages_frame, bg='#151a33' if sender == 'user' else '#111625')
    bubble.pack(fill=tk.X, pady=8, padx=12, anchor='e' if sender == 'user' else 'w')

    message_color = '#5890ff' if sender == 'user' else '#2f3455'
    text_color = '#ffffff' if sender == 'user' else '#d6d9ff'
    align = 'e' if sender == 'user' else 'w'

    label = tk.Label(
        bubble,
        text=text,
        bg=message_color,
        fg=text_color,
        font=('Segoe UI', 11),
        wraplength=420,
        justify=tk.LEFT,
        padx=12,
        pady=10,
        bd=0
    )
    label.pack(anchor=align)

    if sender == 'user':
        bubble.configure(bg='#151a33')
    else:
        bubble.configure(bg='#111625')

    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1.0)


# ---------------- UI DESIGN ----------------
root = tk.Tk()
root.title('🪄 Premium FAQ Chatbot')
root.geometry('980x680')
root.configure(bg='#0b0d1a')
root.minsize(900, 650)

style = ttk.Style(root)
style.theme_use('clam')
style.configure('Accent.TButton', font=('Segoe UI Variable', 12, 'bold'), foreground='#ffffff', background='#5b7dff', borderwidth=0, focusthickness=0, padding=(14, 10))
style.map('Accent.TButton', background=[('active', '#3f5cff'), ('pressed', '#2f52ff')], foreground=[('disabled', '#a3b5ff')])
style.configure('Suggestion.TButton', font=('Segoe UI Variable', 10), foreground='#edf0ff', background='#1f2a5c', borderwidth=0, padding=(10, 8))
style.map('Suggestion.TButton', background=[('active', '#2b3a78'), ('pressed', '#1f2a5c')])
style.configure('Vertical.TScrollbar', gripcount=0, background='#5e75ff', troughcolor='#0c1022', bordercolor='#0c1022', arrowcolor='#e8efff', lightcolor='#5e75ff', darkcolor='#5e75ff')
style.layout('Vertical.TScrollbar', [('Vertical.Scrollbar.trough', {'children': [('Vertical.Scrollbar.thumb', {'unit': '1', 'children': [('Vertical.Scrollbar.grip', {'sticky': ''})], 'sticky': 'nswe'})], 'sticky': 'ns'})])
style.configure('TFrame', background='#0b0d1a')
style.configure('TLabel', background='#0b0d1a', foreground='#edf0ff', font=('Segoe UI Variable', 10))
style.configure('TEntry', fieldbackground='#111627', foreground='#ffffff', bordercolor='#2a2f4b', padding=8)

main_container = tk.Frame(root, bg='#0b0d1a')
main_container.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)

left_panel = tk.Frame(main_container, bg='#13182f')
left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

right_panel = tk.Frame(main_container, bg='#12162f', width=280)
right_panel.pack(side=tk.RIGHT, fill=tk.Y)
right_panel.pack_propagate(False)


header = tk.Frame(left_panel, bg='#161d38', pady=16)
header.pack(fill=tk.X)
logo = tk.Label(header, text='AI Assistant', bg='#161d38', fg='#ffffff', font=('Segoe UI', 16, 'bold'))
logo.pack(side=tk.LEFT, padx=16)
status_dot = tk.Canvas(header, width=10, height=10, bg='#161d38', highlightthickness=0)
status_dot.create_oval(0, 0, 10, 10, fill='#3ee58f', outline='')
status_dot.pack(side=tk.LEFT, padx=(8, 0), pady=4)
status_label = tk.Label(header, text='Online', bg='#161d38', fg='#8aa1ff', font=('Segoe UI', 10, 'normal'))
status_label.pack(side=tk.LEFT, padx=(6, 0))

chat_canvas = tk.Canvas(left_panel, bg='#101429', bd=0, highlightthickness=0)
chat_canvas.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, command=chat_canvas.yview, style='Vertical.TScrollbar')
scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4))
chat_canvas.configure(yscrollcommand=scrollbar.set)

messages_frame = tk.Frame(chat_canvas, bg='#0f1225')
messages_frame.bind('<Configure>', lambda e: chat_canvas.configure(scrollregion=chat_canvas.bbox('all')))
chat_canvas.create_window((0, 0), window=messages_frame, anchor='nw')

welcome = tk.Label(
    messages_frame,
    text='Hello! I\'m your premium FAQ assistant. Ask me anything about orders, shipping, returns, or products.',
    bg='#2c3456',
    fg='#f5f7ff',
    font=('Segoe UI', 11),
    wraplength=420,
    justify=tk.LEFT,
    padx=14,
    pady=14
)
welcome.pack(fill=tk.X, pady=(16, 6), padx=12, anchor='w')

# Preload a sample conversation to match the screenshot style
sample_history = [
    ('Hi, can you help me?', 'user'),
    ('Absolutely! Please tell me what you need help with.', 'bot'),
    ('Do you offer international shipping?', 'user'),
    ('Yes, we ship our products worldwide.', 'bot'),
    ('How long does delivery usually take?', 'user'),
    ('Delivery usually takes 3–7 working days depending on the destination.', 'bot')
]
for message, sender in sample_history:
    add_message(message, sender=sender)

footer = tk.Frame(left_panel, bg='#13182f', pady=14)
footer.pack(fill=tk.X)

entry = ttk.Entry(footer, font=('Segoe UI Variable', 12), foreground='#abb4dc')
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(16, 10))
entry.insert(0, 'Type your question here...')
entry.bind('<Return>', send_message)

send_btn = ttk.Button(footer, text='Send', style='Accent.TButton', command=send_message)
send_btn.pack(side=tk.RIGHT, padx=(0, 16))

typing_label = tk.Label(footer, text='', bg='#13182f', fg='#9bb3ff', font=('Segoe UI Variable', 9, 'italic'))
typing_label.pack(side=tk.BOTTOM, anchor='w', padx=(16, 0), pady=(6, 0))


# Placeholder behavior for the entry field
placeholder_text = 'Type your question here...'
def clear_placeholder(event):
    if entry.get() == placeholder_text:
        entry.delete(0, tk.END)
        entry.config(foreground='#ffffff')

def add_placeholder(event):
    if entry.get().strip() == '':
        entry.insert(0, placeholder_text)
        entry.config(foreground='#abb4dc')

entry.bind('<FocusIn>', clear_placeholder)
entry.bind('<FocusOut>', add_placeholder)

bot_photo_frame = tk.Frame(right_panel, bg='#1c233f', height=200)
bot_photo_frame.pack(fill=tk.X)
bot_photo_frame.pack_propagate(False)

# Try to load an external avatar image 'bot_avatar.png' if present; otherwise draw the default canvas avatar
try:
    avatar_img = tk.PhotoImage(file='bot_avatar.png')
    avatar_label = tk.Label(bot_photo_frame, image=avatar_img, bg='#1c233f')
    avatar_label.image = avatar_img
    avatar_label.pack(pady=16)
except Exception:
    photo_canvas = tk.Canvas(bot_photo_frame, width=120, height=120, bg='#1c233f', highlightthickness=0)
    photo_canvas.create_oval(10, 10, 110, 110, outline='#6b85ff', width=4, fill='#12182f')
    photo_canvas.create_oval(40, 32, 80, 72, fill='#edf0ff', outline='')
    photo_canvas.create_oval(50, 40, 58, 48, fill='#12182f', outline='')
    photo_canvas.create_oval(62, 40, 70, 48, fill='#12182f', outline='')
    photo_canvas.create_arc(40, 58, 80, 90, start=190, extent=160, style='arc', outline='#edf0ff', width=3)
    photo_canvas.create_rectangle(45, 78, 75, 102, fill='#2b2f4f', outline='')
    photo_canvas.create_text(60, 90, text='AI', fill='#7c8cff', font=('Segoe UI', 10, 'bold'))
    photo_canvas.pack(pady=16)

# Assistant display name (updated per request)
bot_name = tk.Label(right_panel, text='EL Make Assistant', bg='#12162f', fg='#ffffff', font=('Segoe UI', 12, 'bold'))
bot_name.pack(anchor='center', pady=(0, 4))
role_label = tk.Label(right_panel, text='Smart FAQ concierge', bg='#12162f', fg='#a3adff', font=('Segoe UI', 9))
role_label.pack(anchor='center')

info_label = tk.Label(right_panel, text='Premium assistant tuned for FAQs and customer support.', bg='#12162f', fg='#b6bde8', wraplength=240, justify=tk.LEFT, font=('Segoe UI', 10))
info_label.pack(padx=18, pady=(14, 12))

divider = tk.Frame(right_panel, bg='#1b2140', height=1)
divider.pack(fill=tk.X, padx=18, pady=(0, 14))

feature_title = tk.Label(right_panel, text='Top Features', bg='#12162f', fg='#ffffff', font=('Segoe UI', 12, 'bold'))
feature_title.pack(anchor='w', padx=18)

features = [
    '• Fast FAQ matching',
    '• NLP preprocessing',
    '• Cosine similarity ranking',
    '• Elegant chat bubble design'
]
for feature in features:
    lbl = tk.Label(right_panel, text=feature, bg='#12162f', fg='#aab2ff', font=('Segoe UI Variable', 10), anchor='w', justify=tk.LEFT)
    lbl.pack(fill=tk.X, padx=24, pady=4)

suggestion_title = tk.Label(right_panel, text='Quick questions', bg='#12162f', fg='#ffffff', font=('Segoe UI Variable', 12, 'bold'))
suggestion_title.pack(anchor='w', padx=18, pady=(18, 8))

suggestions = [
    'What products do you sell?',
    'Do you offer international shipping?',
    'Can I return a product?',
    'How can I contact support?'
]
for question in suggestions:
    def make_command(q=question):
        entry.delete(0, tk.END)
        entry.insert(0, q)
        entry.config(foreground='#ffffff')
        send_message()

    suggestion_btn = ttk.Button(right_panel, text=question, style='Suggestion.TButton', command=make_command)
    suggestion_btn.pack(fill=tk.X, padx=18, pady=4)

bottom_note = tk.Label(right_panel, text='Powered by a premium AI chatbot interface.', bg='#12162f', fg='#7c86ab', wraplength=240, justify=tk.LEFT, font=('Segoe UI Variable', 9))
bottom_note.pack(side=tk.BOTTOM, pady=18, padx=18)

root.mainloop()
