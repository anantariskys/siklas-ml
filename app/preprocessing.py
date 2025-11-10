import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Unduh data NLTK (sekali saja)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Stopwords ID + EN
stop_words = set(stopwords.words("indonesian") + stopwords.words("english"))

# Stemmer & Lemmatizer
stemmer = StemmerFactory().create_stemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1️⃣ Lowercase
    text = text.lower()

    # 2️⃣ Hapus URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # 3️⃣ Hapus angka & tanda baca, tapi simpan huruf a-zA-Z
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 4️⃣ Hapus spasi ganda
    text = re.sub(r"\s+", " ", text).strip()

    # 5️⃣ Tokenisasi
    tokens = text.split()

    # 6️⃣ Hapus stopwords & token pendek
    tokens = [w for w in tokens if w not in stop_words and len(w) >= 3]

    # 7️⃣ Stemming + Lemmatization
    cleaned_tokens = []
    for w in tokens:
        # Lemmatize Inggris
        lemma = lemmatizer.lemmatize(w)
        # Stem Indo
        stemmed = stemmer.stem(lemma)
        cleaned_tokens.append(stemmed)

    # 8️⃣ Gabungkan kembali
    return " ".join(cleaned_tokens)
