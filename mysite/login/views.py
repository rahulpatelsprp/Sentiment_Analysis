from django.http import HttpResponse
from django.shortcuts import render, redirect
from .models import Users
import smtplib
import ssl
from email.message import EmailMessage
import nltk
from googletrans import Translator
import speech_recognition as sr

from keras.models import load_model
import pandas as pd
import numpy as np
import os

import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('vader_lexicon')
nltk.download('stopwords')
# Create your views here.
def index(request):
    return render(request, 'login.html')


def success(request):
    if request.method == 'POST':
        entered_username = request.POST.get('username')
        entered_password = request.POST.get('password')
        Users_registered = Users.objects.filter(username=entered_username, password=entered_password)
        if Users_registered.exists():
            return redirect('choice')
        else:
            return redirect("index")
    return redirect("index")


def signup(request):
    return render(request, 'signup.html')


def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        Email = request.POST.get('email')
        Login = Users(username=username, mail=Email, password=password)
        try:
            Login.save()
        except:
            return render(request, 'error.html')
    return render(request, 'login.html')


def forgot_password(request):
    return render(request, 'forgot_password.html')


def send_password(request):
    if request.method == 'POST':
        entered_username = request.POST.get('username')
        entered_email = request.POST.get('email')
        Users_registered = Users.objects.filter(username=entered_username, mail=entered_email)
        if Users_registered.exists():
            ob = Users.objects.get(username=entered_username)
            password_tosend = ob.password
            try:
                email_sender = 'rashijain1710@gmail.com'
                email_password = 'gyri qyfm hque magq'
                email_receiver = entered_email
                em = EmailMessage()
                em['From'] = email_sender
                em['To'] = email_receiver
                em['Subject'] = "Welcome to sentiment analysis"
                em.set_content(str(password_tosend))
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                    smtp.login(email_sender, email_password)
                    smtp.sendmail(email_sender, email_receiver, em.as_string())
            except:
                return redirect("home")


        else:
            return redirect("home")
    return redirect("index")


def choice(request):
    return render(request, 'choice.html')


def AudioBased(request):
    return render(request, 'AudioBased.html')


def TextBased(request):
    return render(request, 'TextBased.html')

def error(request):
    return render(request, 'error.html')

def ResultAudio(request):
    return HttpResponse('Audio')
def model_training():
    global train_path,ps,cv,data,model1,cv_transformer
    train_path = pd.read_csv(os.getcwd() +'\\login\\train.csv', encoding='ISO-8859-1', nrows=10000)
    ps = PorterStemmer()
    cv = CountVectorizer()
    train_path = train_path.dropna(subset=['text'])
    data = []  # list
    for i in range(0, min(27480, len(train_path))):
        review = train_path['text'].iloc[i]  # using iloc to access rows by integer position
        review = re.sub('[^a-zA-Z]', ' ', review)  # removing special characters
        review = review.lower()
        review = review.split()
        review = [ps.stem(w) for w in review if w not in set(stopwords.words('english'))]  # stemming & removing stopwords
        review = ' '.join(review)  # joining words
        data.append(review)  # for saving the iteration(preprocess data)
    # x = cv.fit_transform(data).toarray()
    # y = cv.fit_transform(train_path['sentiment']).toarray()
    
    # X_train,X_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 42)
    
    # model1 = Sequential()
    # model1.add(Dense(5000,activation ='relu'))
    # model1.add(Dense(100, activation='relu'))
    # model1.add(Dense(3,activation ='sigmoid'))
    # model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model1.fit(X_train, y_train, epochs=2)
    # model1.save('sentiments.h5')
    # evaluation = model1.evaluate(X_test, y_test)
    cv_transformer = cv.fit(data)


def predict_sentiment(new_review):
    model_training()
    model=load_model('sentiments.h5')
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
    new_review = ' '.join(new_review)
    new_X_test = cv_transformer.transform([new_review]).toarray()
    new_y_pred = model.predict(new_X_test)
    return new_y_pred


def ResultText(request):
    if request.method == 'POST':
        choosen_language = request.POST.get('select_box')
        if choosen_language == 'en':
            text = request.POST.get('text_input')
            new_review_prediction = predict_sentiment(text)
            predicted_class = np.argmax(new_review_prediction)
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            predicted_sentiment = sentiment_labels[predicted_class]
            return render(request, 'Result.html', {'Entered_text': text, 'Sentiment': predicted_sentiment, })

        else:

            lan_text = request.POST.get('text_input')

        # Download the VADER lexicon (if not already downloaded)


            translator = Translator()
            text = translator.translate(lan_text, src=choosen_language, dest='en').text
        # Perform sentiment analysis
            new_review_prediction = predict_sentiment(text)
            predicted_class = np.argmax(new_review_prediction)
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            predicted_sentiment = sentiment_labels[predicted_class]
            return render(request, 'Result.html', {'Entered_text': lan_text, 'Sentiment': predicted_sentiment, })

    return redirect('index')


def Result(request):
    return render(request, 'Result.html')

def listen(request):
    global audio_record
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        try:
            audio_record=r.recognize_google(audio)
        except Exception as e:
            audio_record=''
    new_review_prediction = predict_sentiment(audio_record)
    predicted_class = np.argmax(new_review_prediction)
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[predicted_class]
    return render(request, 'Result.html', {'Entered_text': audio_record, 'Sentiment': predicted_sentiment, })

