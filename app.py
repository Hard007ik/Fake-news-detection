import streamlit as st

# To Load MLPClassifier Model
import joblib
mlp_model = joblib.load('Fake_News_Detection_MLPClassifier_Model.pkl')
tv = joblib.load('tfidf_vectorizer.pkl')

# To load LSTM Model
from tensorflow.keras.models import load_model  # type: ignore
lstm_model = load_model('Fake_News_Detection_LSTM_MODEL.h5')
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore

# print('Model loaded successfull')
def preprocessing(news):
    
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    news = news.lower()

#  remove_url takes raw text and removes urls from the text.
    news = re.sub(r"http\S+", "", news)


    # Load stopwords
    stop = stopwords.words('english')

    # Perform cleaning operations on the first element of the list (text[0])
    news = re.sub(r'[^A-Za-z0-9\s]', '', news)  # Remove non-alphanumeric characters using regex
    news = news.replace('\n', '')  # Replace newlines with spaces
    news = re.sub(r'\s+', ' ', news)  # Replace multiple spaces with a single space

# Remove stopwords
    news = " ".join([word for word in news.split() if word.lower() not in stop])

    def lemmatization(words):
        wl = WordNetLemmatizer()
        lemma_words = []
        for i in words:
        # applying Lemmatization
            lemma_words.append(wl.lemmatize(i))

        return " ".join(lemma_words)

# converting text into tokens of word like 'donald trump wish american' to ['donald', 'trump', 'wish', 'american']
    tokens = word_tokenize(news)
    news = lemmatization(tokens)

    return news



# title
st.title("Fake News Detection")

news  = st.text_area("Enter News", 'Enter here...')
news = preprocessing(news)

st.text("Choose Model: ")
if st.button("MLPClassifier"):
    x_test = tv.transform([news])
    y_predict = mlp_model.predict(x_test)
    answer = "Fake" if y_predict[0]<0.5 else "Real"
    st.success(answer)
if st.button("LSTM Model"):
    tokenizer = Tokenizer()
    # convert data-text into single word
    tokenizer.fit_on_texts([news])
    # word_index(Vocabulary of corpus) a dictionary with word and their index
    word_index = tokenizer.word_index
    # print('Total Unique words are ',len(word_index))

    # padding data to make all text rows with same length
    sequences = tokenizer.texts_to_sequences([news])
    # padding='post' - to make same length text(news) adding extra 0 at end of the text
    # truncating='post' -(default) after making len of 10000 truncate those words whose has more len than 10000 at end
    test = pad_sequences(sequences, maxlen=10000, padding='post')

    prediction = lstm_model.predict(test)
    answer = "Fake" if prediction[0][0]<0.5 else "Real"

    st.success(answer)