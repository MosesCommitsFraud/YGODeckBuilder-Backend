import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Preprocess text function (tokenization, stopword removal, lemmatization)
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Lowercase, remove special characters and numbers
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return ' '.join(tokens)


# Load and preprocess card descriptions and names
def load_and_preprocess_card_data(card_data_file):
    with open(card_data_file, 'r') as file:
        card_data = json.load(file)

    descriptions = {}
    names = {}

    for card in card_data['data']:
        card_name = card['name']
        card_description = card.get('desc', "")
        if card_description:
            descriptions[card_name] = preprocess_text(card_description)
        names[card_name] = preprocess_text(card_name)

    return descriptions, names


# Extract frequent terms from descriptions
def extract_description_keywords(descriptions):
    count_vectorizer = CountVectorizer(max_df=0.85, min_df=2, ngram_range=(1, 3))
    count_matrix = count_vectorizer.fit_transform(descriptions.values())

    # Extract top keywords from descriptions
    terms = count_vectorizer.get_feature_names_out()

    return count_matrix, terms


# Perform topic modeling (LDA) to find common themes/synergies
def perform_topic_modeling(count_matrix, terms, n_topics=10):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(count_matrix)

    # Extract and display top words in each topic
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [terms[i] for i in topic.argsort()[:-11:-1]]
        topics[f"Topic {topic_idx}"] = top_words

    return topics


# Extract name-based patterns (n-grams)
def extract_name_keywords(names, n=1):
    ngrams = {}
    for name, preprocessed_name in names.items():
        tokens = preprocessed_name.split()
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i + n])
            if ngram in ngrams:
                ngrams[ngram].append(name)
            else:
                ngrams[ngram] = [name]
    return ngrams


# Combine description and name keywords for each card
def combine_keywords(descriptions, terms, name_1grams, name_2grams):
    keywords_per_card = {}

    for card, description in descriptions.items():
        # Get top keywords from description
        keywords = [term for term in description.split() if term in terms]

        # Add name-based n-grams
        name_keywords = []
        for ngram, cards in name_1grams.items():
            if card in cards:
                name_keywords.append(ngram)
        for ngram, cards in name_2grams.items():
            if card in cards:
                name_keywords.append(ngram)

        # Combine keywords from description and name
        combined_keywords = list(set(keywords + name_keywords))
        keywords_per_card[card] = combined_keywords

    return keywords_per_card


# Main function to perform analysis and save results
def analyze_cards(card_data_file, output_json='card_keywords_auto.json', output_csv='cards_keywords_matrix_auto.csv'):
    descriptions, names = load_and_preprocess_card_data(card_data_file)

    # Extract description keywords using CountVectorizer
    count_matrix, terms = extract_description_keywords(descriptions)

    # Perform LDA topic modeling
    topics = perform_topic_modeling(count_matrix, terms)
    print("Identified Topics and Synergy Keywords:")
    for topic, words in topics.items():
        print(f"{topic}: {', '.join(words)}")

    # Extract name-based 1-gram and 2-gram patterns
    name_1grams = extract_name_keywords(names, n=1)
    name_2grams = extract_name_keywords(names, n=2)

    # Combine description and name keywords
    keywords_per_card = combine_keywords(descriptions, terms, name_1grams, name_2grams)

    # Save keywords for each card in JSON format
    with open(output_json, 'w') as file:
        json.dump(keywords_per_card, file)

    # Prepare dataset for modeling (cards with keyword vectors)
    cards_keywords_matrix = pd.DataFrame(columns=terms, index=descriptions.keys())
    for card in descriptions.keys():
        keyword_vector = [1 if term in keywords_per_card[card] else 0 for term in terms]
        cards_keywords_matrix.loc[card] = keyword_vector

    # Save the model-ready data (cards with keyword vectors)
    cards_keywords_matrix.to_csv(output_csv)

    print(f"Analysis complete. Results saved to {output_json} and {output_csv}")
