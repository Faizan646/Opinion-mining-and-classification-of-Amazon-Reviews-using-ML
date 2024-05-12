import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from keras.utils import to_categorical
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Download NLTK resources
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")


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

labeled_reviews_df = pd.read_csv("IMDB Dataset.csv")
labeled_reviews_df["content"] = labeled_reviews_df["review"].apply(clean_text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(labeled_reviews_df["content"])
X_text = tokenizer.texts_to_sequences(labeled_reviews_df["content"])
X_text = pad_sequences(X_text, maxlen=200)  

label_encoder = LabelEncoder()
labeled_reviews_df["sentiment_encoded"] = label_encoder.fit_transform(labeled_reviews_df["sentiment"])

y_encoded = labeled_reviews_df["sentiment_encoded"] 

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y_encoded, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# RNN
rnn_model = Sequential()
rnn_model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=200, input_length=200))
rnn_model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))  # Adding dropout
rnn_model.add(Dense(3, activation='softmax'))
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.fit(X_train, y_train, epochs=10, batch_size=512, validation_data=(X_test, y_test))
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test)
print("RNN Model Accuracy:", rnn_accuracy)
rnn_model.save("rnn_model.h5")

# CNN
# cnn_model = Sequential()
# cnn_model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=200, input_length=200))
# cnn_model.add(Conv1D(128, 5, activation='relu'))
# cnn_model.add(MaxPooling1D(5))
# cnn_model.add(Conv1D(128, 5, activation='relu'))
# cnn_model.add(MaxPooling1D(5))
# cnn_model.add(Flatten())
# cnn_model.add(Dense(128, activation='relu'))
# cnn_model.add(Dropout(0.5))
# cnn_model.add(Dense(3, activation='softmax'))  
# cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
# cnn_model.fit(X_train, y_train, epochs=10, batch_size=512, validation_data=(X_test, y_test))
# cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)
# print("CNN Model Accuracy:", cnn_accuracy)
# cnn_model.save("cnn_model.h5")

#DNN
dnn_model = Sequential()
dnn_model.add(Dense(128, activation='relu', input_dim=200))
dnn_model.add(Dropout(0.5))
dnn_model.add(Dense(64, activation='relu'))
dnn_model.add(Dropout(0.5))
dnn_model.add(Dense(3, activation='softmax'))  # Three output units for three classes
dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
dnn_model.fit(X_train, y_train, epochs=10, batch_size=512, validation_data=(X_test, y_test))
dnn_loss, dnn_accuracy = dnn_model.evaluate(X_test, y_test)
print("DNN Model Accuracy:", dnn_accuracy)
dnn_model.save("dnn_model.h5")