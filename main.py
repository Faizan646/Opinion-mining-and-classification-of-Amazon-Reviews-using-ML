from flask import Flask, request, render_template, jsonify
from amazoncaptcha import AmazonCaptcha
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
import random
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")

# Function to clean text
def clean_text(text):
    if text is None:
        return "" 
    text = re.sub(r"<.*?>", "", text) 
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)  
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words] 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    clean_text = " ".join(tokens) 
    return clean_text

def solve_captcha(webdriver):
    captcha_image_element = webdriver.find_element(By.XPATH, "//div[@class = 'a-row a-text-center']//img")
    url_img_captcha = captcha_image_element.get_attribute("src")
    captcha = AmazonCaptcha.fromlink(url_img_captcha)
    captcha_text = captcha.solve()
    captcha_input = webdriver.find_element(By.XPATH, "//input[@id='captchacharacters']")
    captcha_input.send_keys(captcha_text)
    captcha_input.send_keys(Keys.ENTER)
    page_source = webdriver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    return soup

def fetch_page_soup(url):
    options = Options()
    options.headless = True
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    captcha_form_present = is_captcha_form_present(driver)
    if captcha_form_present:
        solve_captcha(driver)
    content = driver.page_source
    soup = BeautifulSoup(content, 'html.parser')
    return soup

def is_captcha_form_present(driver):
    try:
        driver.find_element(By.XPATH, "//form[@action='/errors/validateCaptcha']")
        return True
    except Exception as e:
        return False

def get_total_reviews(url):
    soup = fetch_page_soup(url)
    review_count_text = soup.find('span', {'id': 'acrCustomerReviewText'}).get_text(strip=True).replace(',', '')
    if "rating" in review_count_text:
        review_count_text = review_count_text.split(" ")[0]
    total_review_count = int(review_count_text)
    return total_review_count

def fetch_reviews(products_url, total_review_count, cnn_model, tokenizer):
    reviews = []
    expected_review_pages = int(total_review_count / 10)
    for i in range(expected_review_pages):
        try:
            print("Fetching page:", products_url)  # Print the URL being fetched
            soup = fetch_page_soup(products_url)
        except Exception as e:
            print("Exception occurred while fetching page:", e)  # Print any exception that occurs
            continue  # Continue to the next iteration if there's an exception
            
        review_elements = soup.find_all('div', {'class': 'a-section review'})
        if review_elements == []:
            review_elements = soup.find_all('div', {'class': 'a-section review aok-relative'})
            
        print("review_elements:", review_elements)  # Print the review elements found
        for element in review_elements:
            r_author_element = element.find('span', {'class': 'a-profile-name'})
            r_author = r_author_element.text.strip() if r_author_element else None
            r_date_element = element.find('span', {'data-hook': 'review-date'})
            r_date = r_date_element.text.strip() if r_date_element else None
            r_title_element = element.find('a', {'class': 'a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold'})
            r_title = r_title_element.text.strip() + " This was a gift" if r_title_element else None
            r_content_element = element.find('span', {'data-hook': 'review-body'})
            r_content = r_content_element.text.strip() if r_content_element else None
            r_rating_element = element.find('i', {'data-hook': 'review-star-rating'})
            r_rating = float(r_rating_element.text.split(' ')[0]) if r_rating_element else None
            r_verified_element = element.find('span', {'data-hook': 'avp-badge'})
            r_verified = True if r_verified_element else False
            
            # Clean and tokenize review body and title
            cleaned_text = clean_text(r_content)
            X_text = tokenizer.texts_to_sequences([cleaned_text])
            X_text = pad_sequences(X_text, maxlen=200)
            
            cleaned_title = clean_text(r_title)
            X_heading = tokenizer.texts_to_sequences([cleaned_title])
            X_heading = pad_sequences(X_heading, maxlen=200)
            
            # Predict sentiment for review body and title
            body_sentiment_cnn = predict_sentiment(cnn_model, tokenizer, X_text)
            heading_sentiment_cnn = predict_sentiment(cnn_model, tokenizer, X_heading)
            
            # Calculate final opinion based on rating and sentiments
            final_opinion_cnn = calculate_final_opinion(r_rating, heading_sentiment_cnn, body_sentiment_cnn)
            
            # Construct review dictionary
            review = {
                "author": r_author,
                "date": r_date,
                "heading": r_title,
                "content": r_content,
                "rating": r_rating,
                "verified": r_verified,
                "body_sentiment_cnn": body_sentiment_cnn,
                "heading_sentiment_cnn": heading_sentiment_cnn,
                "final_opinion_cnn": final_opinion_cnn
            }
            reviews.append(review)
        try:
            next_button = soup.find('li', {'class': 'a-last'}).find('a')
            products_url = "https://www.amazon.com" + next_button['href']
        except Exception as e:
            print("Exception:", e)
            break

    reviews_df = pd.DataFrame(reviews)
    return reviews_df


def predict_sentiment(model, tokenizer, X_text):
    predicted_sentiment = model.predict(X_text)[0]
    if predicted_sentiment.argmax() == 0:
        sentiment_label = "negative"
    elif predicted_sentiment.argmax() == 1:
        sentiment_label = "positive"
    return sentiment_label

def calculate_final_opinion(rating, heading_sentiment, body_sentiment):
    switch = {
        5: lambda: 'positive' if heading_sentiment == 'positive' or body_sentiment == 'positive' else 'neutral',
        4: lambda: 'positive' if heading_sentiment == 'positive' or body_sentiment == 'positive' else 'neutral',
        3: lambda: 'neutral' if (heading_sentiment != 'negative' or body_sentiment != 'negative') else ('negatuve' if heading_sentiment == 'negative' and body_sentiment == 'negative' else 'positive'),
        2: lambda: 'negative' if heading_sentiment == 'negative' or body_sentiment == 'negative' else 'neutral',
        1: lambda: 'negative' if heading_sentiment == 'negative' or body_sentiment == 'negative' else 'neutral'
    }
    
    return switch.get(rating, lambda: 'N/A')()




def generate_first_reviews_url(product_url):
    base_url, path = product_url.split("/dp/")
    asin = path.split("/")[0]
    all_reviews_url = f"{base_url}/product-reviews/{asin}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber=1"
    return all_reviews_url

def trigger_fetch_reviews_procedure(url):
    total_review_count = get_total_reviews(url)
    first_page_url = generate_first_reviews_url(url)
    reviews = fetch_reviews(first_page_url, total_review_count,cnn_model, tokenizer)
    return reviews

app = Flask(__name__)
cnn_model = load_model("cnn_model.h5")
tokenizer = joblib.load("tokenizer.joblib")

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/get_total_reviews', methods=['POST'])
def get_total_reviews_route():
    url = request.form.get('url')
    reviews_df = trigger_fetch_reviews_procedure(url)
    print(reviews_df)
    print(reviews_df.columns)
    reviews_df.to_csv("reviews_with_sentiments.csv", index=False)
    reviews_json = reviews_df.to_dict(orient='records')
    return jsonify(reviews_json)

if __name__ == '__main__':
    app.run(debug=True)
