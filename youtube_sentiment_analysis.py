from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from googleapiclient.discovery import build
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def sentiment(comment):
    inputs = tokenizer(comment, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    pred_class = torch.argmax(outputs.logits, dim=1).item()
    
    if pred_class == 0:
        return("Negative Comment")
    else:
        return("Positive Comment")


def generate_sentiment_analysis(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None
    
    while True:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100, 
            pageToken=next_page_token if next_page_token else ''
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        next_page_token = response.get('nextPageToken')

        if not next_page_token:
            break
            
    df = pd.DataFrame(np.array(comments), columns=['comments'])
    df['sentiment'] = df['comments'].apply(lambda x: sentiment(x[:512]))
    df.to_csv('despacito_comments.csv', index=False)

    sent_counts = df['sentiment'].value_counts(normalize=True)
    sent_counts.plot(kind='pie', autopct='%1.0f%%', colors=['pink', 'silver'], explode=(0.05, 0.05))
    plt.title('Sentiment Distribution')
    plt.ylabel('')
    plt.savefig('sentiment_distribution.png')
    plt.show()


    pos_words = ' '.join(df[df['sentiment'] == 'Positive Comment']['comments'])
    neg_words = ' '.join(df[df['sentiment'] == 'Negative Comment']['comments'])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(pos_words)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Positive Sentiments Word Cloud')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(neg_words)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Negative Sentiments Word Cloud')
    plt.axis('off')
    plt.savefig('word_cloud.png')
    plt.show()



if __name__ == "__main__":
    api_key = 'your_api_key'  # Replace with your actual YouTube API key
    video_id = 'your_video_id'  # Replace with the desired video's ID

    # Load tokenizer and model
    gpt2 = 'michelecafagna26/gpt2-medium-finetuned-sst2-sentiment'
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2)
    model = GPT2ForSequenceClassification.from_pretrained(gpt2)

    generate_sentiment_analysis(api_key, video_id)
               




        




