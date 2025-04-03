import re
from collections import Counter
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pypdf
import numpy as np

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')

def get_synonyms(word):
    """Returns a set of synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms

def extract_text(file_path):
    """Extracts text content from a .txt or .pdf file."""
    try:
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        elif file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ''.join([page.extract_text() or '' for page in reader.pages])
        else:
            return "Unsupported file format. Please provide a .txt or .pdf file."
        return text
    except FileNotFoundError:
        return "File not found. Please check the file path."
    except Exception as e:
        return f"Error processing file: {str(e)}"

def preprocess_text(text):
    """Cleans and preprocesses text using spaCy for better tokenization."""
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def extract_key_phrases(text):
    """Extracts key phrases (noun chunks) from text using spaCy."""
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

def calculate_similarity(resume_text, job_description_text):
    """Calculates similarity using TF-IDF with synonym enhancement."""
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description_text])
    similarity = cosine_similarity(vectors)[0][1]
    return similarity

def analyze_keywords(resume_text, job_description_text, top_n=10):
    """Analyzes keywords with synonym matching and weights."""
    resume_doc = nlp(resume_text)
    job_doc = nlp(job_description_text)
    
    # Extract single words and phrases
    resume_words = [token.text for token in resume_doc if token.is_alpha and not token.is_stop]
    job_words = [token.text for token in job_doc if token.is_alpha and not token.is_stop]
    resume_phrases = extract_key_phrases(resume_text)
    job_phrases = extract_key_phrases(job_description_text)

    # Build synonym-enhanced keyword sets
    job_keywords = set(job_words + job_phrases)
    resume_keywords = set(resume_words + resume_phrases)
    job_synonyms = {word: get_synonyms(word) for word in job_keywords}

    # Count matches, including synonyms
    resume_counts = Counter(resume_words + resume_phrases)
    job_counts = Counter(job_words + job_phrases)
    common_keywords = set()
    synonym_matches = set()

    for res_word in resume_keywords:
        if res_word in job_keywords:
            common_keywords.add(res_word)
        else:
            for job_word, syns in job_synonyms.items():
                if res_word in syns:
                    synonym_matches.add((res_word, job_word))

    return {
        "resume_keywords": resume_counts.most_common(top_n),
        "job_keywords": job_counts.most_common(top_n),
        "common_keywords": common_keywords,
        "synonym_matches": synonym_matches
    }

def detect_experience(text):
    """Detects years of experience using regex."""
    pattern = r'(\d+\+?\s*years?\s*(?:experience|exp))'
    matches = re.findall(pattern, text.lower())
    return matches

def ats_check(resume_path, job_description_path):
    """Performs an ATS-like check with modern features."""
    resume_text = extract_text(resume_path)
    if "Error" in resume_text or "Unsupported" in resume_text or "not found" in resume_text:
        print("Unsupported resume")
        return resume_text
    job_description_text = extract_text(job_description_path)
    if "Error" in job_description_text or "Unsupported" in job_description_text or "not found" in job_description_text:
        print("Unsupported jd")
        return job_description_text

    # Preprocess text
    processed_resume = preprocess_text(resume_text)
    processed_job = preprocess_text(job_description_text)

    # Calculate similarity
    similarity_score = calculate_similarity(processed_resume, processed_job)
    
    # Analyze keywords and phrases
    keyword_analysis = analyze_keywords(processed_resume, processed_job)
    
    # Detect experience
    resume_experience = detect_experience(resume_text)
    job_experience = detect_experience(job_description_text)

    # Dynamic scoring based on job requirements
    score_adjustment = 0
    if job_experience and not resume_experience:
        score_adjustment -= 0.2  # Penalty for missing experience
    elif job_experience and resume_experience:
        job_years = int(re.search(r'\d+', job_experience[0]).group()) if job_experience else 0
        res_years = int(re.search(r'\d+', resume_experience[0]).group()) if resume_experience else 0
        if res_years < job_years:
            score_adjustment -= 0.1  # Penalty for insufficient experience

    final_score = max(0, min(1, similarity_score + score_adjustment))

    # Output results
    print(f"Similarity Score: {final_score:.2f}")
    print("\nKeyword Analysis:")
    print(f"  Top Resume Keywords: {keyword_analysis['resume_keywords']}")
    print(f"  Top Job Description Keywords: {keyword_analysis['job_keywords']}")
    print(f"  Common Keywords: {keyword_analysis['common_keywords']}")
    if keyword_analysis['synonym_matches']:
        print(f"  Synonym Matches: {keyword_analysis['synonym_matches']}")
    
    print("\nExperience Check:")
    print(f"  Resume Experience: {resume_experience if resume_experience else 'None detected'}")
    print(f"  Job Requirement: {job_experience if job_experience else 'None specified'}")

    # Feedback based on dynamic thresholds
    if final_score < 0.4:
        print("\nLow match. Tailor your resume with more relevant keywords, phrases, and experience.")
    elif final_score < 0.7:
        print("\nModerate match. Enhance keyword alignment and ensure experience meets requirements.")
    else:
        print("\nStrong match. Your resume aligns well with the job description.")

if __name__ == "__main__":
    resume_file = input("Enter the path to your resume file (.txt or .pdf): ")
    job_description_file = input("Enter the path to the job description file (.txt or .pdf): ")
    ats_check(resume_file, job_description_file)