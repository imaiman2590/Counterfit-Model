#-------------------------------------- Counterfit Model -----------------------------------------------------
import os
import re
import io
from cryptography.fernet import Fernet
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytesseract
import spacy
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import MarianMTModel, MarianTokenizer, BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from torch.utils.tensorboard import SummaryWriter
from langdetect import detect
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
import albumentations as A
from transformers import MarianMTModel, MarianTokenizer
import torch.onnx
import ipywidgets as widgets
from IPython.display import display

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Constants
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}

# Function to dynamically load a MarianMT model for any given language pair
def load_translation_model(source_lang, target_lang):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

# Initialize translation model (English <-> French as an example, can be switched dynamically)
source_lang = 'en'
target_lang = 'fr'
model, tokenizer = load_translation_model(source_lang, target_lang)

# IMAGE FUNCTIONS
def import_image(folder_path, filename, show=False):
    image_path = os.path.join(folder_path, filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(f"Unsupported image format: {ext}. Supported formats are: {SUPPORTED_IMAGE_FORMATS}")
    image = Image.open(image_path)
    if show:
        image.show()
    return image

def preprocessing_image(image, size=(224, 224)):  
    image = np.array(image)
    resize_image = cv2.resize(image, size)
    color_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2YUV)
    img = color_image.astype('float32') / 255.0
    return img

def image_augmentation(image):
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomSizedCrop(min_max_height=(200, 300), height=224, width=224, p=1),
    ])
    augmented = transform(image=image)
    return augmented['image']

def import_file(file_path):
    ext = str(file_path).lower()
    if ext.endswith('.csv'):
        return pd.read_csv(file_path)
    elif ext.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif ext.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format.")

def extract_image(img):
    return pytesseract.image_to_string(img).strip()

# TEXT FUNCTIONS
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())  # Adjusted regex for clarity
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]  # Ignore punctuation as well
    return " ".join(tokens)

def perform_ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def preprocess(text):
    tf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
    transform_text = tf.fit_transform(text)
    return transform_text

def back_translate(text, model, tokenizer, source_lang, target_lang):
    # Step 1: Translate text to the target language
    translated = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_output = model.generate(**translated)
    translated_text = tokenizer.decode(translated_output[0], skip_special_tokens=True)

    # Step 2: Translate the target language text back to the source language
    back_translated = tokenizer(translated_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    back_translated_output = model.generate(**back_translated)
    final_text = tokenizer.decode(back_translated_output[0], skip_special_tokens=True)

    return final_text

def augment_text_with_back_translation(text, source_lang, target_lang):
    augmented_text = back_translate(text, model, tokenizer, source_lang, target_lang)
    return augmented_text

def process_and_split_images(image_folder, labels):
    image_data = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith(tuple(SUPPORTED_IMAGE_FORMATS)):
            img = import_image(image_folder, filename)
            processed_img = preprocessing_image(img)
            augmented_img = image_augmentation(processed_img)
            image_data.append(augmented_img)
    X = np.array(image_data)
    y = np.array(labels)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# IMAGE MODEL
def fine_tune_model(model, tokenizer, train_data, train_labels, epochs=3, batch_size=16):
    model.train()
    for epoch in range(epochs):
        pass

def image_model(x_train, y_train, x_test, y_test):
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet'))  # Adjusted size
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))  # Binary output
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=32)
    model.evaluate(x_test, y_test)

# BERT TEXT MODEL
def train_text_model(texts, labels, source_lang, target_lang):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Augment text before encoding (if augmentation is desired)
    augmented_texts = [augment_text_with_back_translation(t, source_lang, target_lang) for t in texts]
    encodings = tokenizer(augmented_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    labels_tensor = torch.tensor(labels)
    model.train()
    outputs = model(**encodings, labels=labels_tensor)
    outputs.loss.backward()

def predict_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=-1).item()

# SENTIMENT ANALYSIS
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# FAKE REVIEW DETECTION CLASS
class FakeReviewDetector:
    def __init__(self, model_type="gradient_boosting"):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None

    def _get_model_instance(self):
        model_map = {
            "gradient_boosting": GradientBoostingClassifier(),
            "random_forest": RandomForestClassifier(),
            "logistic_regression": LogisticRegression(max_iter=1000),
            "naive_bayes": MultinomialNB(),
        }
        return model_map[self.model_type]

    def train(self, reviews, labels):
        self.vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
        X = self.vectorizer.fit_transform(reviews)
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

        model = self._get_model_instance()
        model.fit(X_train, y_train)
        self.model = model

        predictions = self.model.predict(X_test)
        print("Classification Report:\n", classification_report(y_test, predictions))

    def predict(self, review):
        if not self.model or not self.vectorizer:
            raise ValueError("Model not trained or loaded.")
        if not isinstance(review, str) or not review.strip():
            raise ValueError("Review must be a non-empty string.")
        X = self.vectorizer.transform([review])
        pred = self.model.predict(X)[0]
        return {"is_fake": bool(pred)}

# ENCRYPTION FUNCTIONS
def encrypt_data(data, key):
    cipher = Fernet(key)
    encrypted_data = cipher.encrypt(data.encode())
    return encrypted_data

# MODEL EXPORT FOR DEPLOYMENT
def export_model_to_onnx(model, example_input, filename='model.onnx'):
    torch.onnx.export(model, example_input, filename)
    print(f"Model saved as {filename}")

# Now, let's add ipywidgets to interact with the user

# Define image upload widget
image_upload = widgets.FileUpload(
    accept='.jpg, .jpeg, .png, .gif, .webp, .bmp, .tiff',
    multiple=False
)

# Display the widget for image upload
display(image_upload)

# Function to process and display the image
def on_image_upload(change):
    uploaded_image = image_upload.value
    if uploaded_image:
        file_info = uploaded_image[list(uploaded_image.keys())[0]]
        filename = file_info['name']
        image_data = file_info['content']
        img = Image.open(io.BytesIO(image_data))
        img.show()

# Add an event listener to the upload widget
image_upload.observe(on_image_upload, names='value')
