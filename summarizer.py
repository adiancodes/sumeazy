from newspaper import Article
import nltk
import pickle
from langdetect import detect
from textblob import TextBlob
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi

# HuggingFace Transformers for local summarization
from transformers import pipeline

# Load summarization pipeline once (BART is good for English, can swap to T5/Pegasus)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Download NLTK data (only if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load tokenizers once at module level
with open('output/punkt_hindi_tokenizer.pkl', 'rb') as f:
    hindi_tokenizer = pickle.load(f)
with open('output/punkt_news_tokenizer.pkl', 'rb') as f:
    english_tokenizer = pickle.load(f)

def summarize_article(url):
    article = Article(url)
    try:
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        raise RuntimeError(f"Error processing article: {e}")

    lang = detect(article.text)
    tokenizer = hindi_tokenizer if lang == 'hi' else english_tokenizer
    lang_name = "Hindi" if lang == 'hi' else "English"

    try:
        sentences = tokenizer.tokenize(article.text)
    except Exception:
        sentences = []

    sentiment = TextBlob(article.text).sentiment

    return {
        "title": article.title,
        "authors": ', '.join(article.authors),
        "date": str(article.publish_date) if article.publish_date else "",
        "language": lang_name,
        "summary": article.summary,
        "polarity": sentiment.polarity,
        "sentiment": "Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral",
        "sentences": sentences
    }

# For BART, max input is 1024 tokens (~1024-1500 words, ~4000 chars). We'll chunk if needed.
MAX_CHARS = 4000

def summarize_youtube_video(youtube_url):
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", youtube_url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    video_id = match.group(1)

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([x['text'] for x in transcript_list])
        if not transcript.strip():
            raise RuntimeError("Sorry, this video does not have a transcript available for summarization.")
    except Exception:
        raise RuntimeError("Sorry, this video does not have a transcript available for summarization.")

    # Chunk transcript if too long for model
    chunks = [transcript[i:i+MAX_CHARS] for i in range(0, len(transcript), MAX_CHARS)]
    summary = ""
    for chunk in chunks:
        # BART expects at least 30 words, so skip tiny chunks
        if len(chunk.split()) < 30:
            continue
        try:
            chunk_summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            summary += chunk_summary + " "
        except Exception as e:
            summary += ""  # Skip chunk if summarization fails

    if not summary.strip():
        summary = "Transcript is too short or could not be summarized."

    lang = detect(transcript)
    lang_name = "Hindi" if lang == 'hi' else "English"
    sentiment = TextBlob(transcript).sentiment
    sentences = nltk.sent_tokenize(transcript)

    return {
        "title": f"YouTube Video ({video_id})",
        "authors": "",
        "date": "",
        "language": lang_name,
        "summary": summary.strip(),
        "polarity": sentiment.polarity,
        "sentiment": "Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral",
        "sentences": sentences
    }
