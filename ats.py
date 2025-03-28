import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(file_path):
    """Extracts text content from a .txt or .pdf file."""
    try:
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        elif file_path.lower().endswith('.pdf'):
            try:
                import pypdf
                with open(file_path, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    text = ''.join([page.extract_text() for page in reader.pages])
            except ImportError:
                return "pypdf is not installed. Please install it using 'pip install pypdf'."
        else:
            return "Unsupported file format. Please provide a .txt or .pdf file."
        return text
    except FileNotFoundError:
        return "File not found. Please check the file path."

def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def calculate_similarity(resume_text, job_description_text):
    """Calculates the cosine similarity between the resume and job description."""
    vectorizer = CountVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description_text])
    similarity = cosine_similarity(vectors)[0][1]
    return similarity

def analyze_keywords(resume_text, job_description_text):
     """Analyzes and compares keywords between the resume and job description."""
     resume_words = resume_text.split()
     job_description_words = job_description_text.split()
    
     resume_word_counts = Counter(resume_words)
     job_description_word_counts = Counter(job_description_words)
    
     common_keywords = set(resume_words) & set(job_description_words)
    
     return {
         "resume_keywords": resume_word_counts,
         "job_description_keywords": job_description_word_counts,
         "common_keywords": common_keywords
     }

def ats_check(resume_path, job_description_path):
    """Performs ATS check and provides feedback."""
    resume_text = extract_text(resume_path)
    if "not installed" in resume_text or "Unsupported" in resume_text or "File not found" in resume_text:
        return resume_text
    job_description_text = extract_text(job_description_path)
    if "not installed" in job_description_text or "Unsupported" in job_description_text or "File not found" in job_description_text:
        return job_description_text

    processed_resume_text = preprocess_text(resume_text)
    processed_job_description_text = preprocess_text(job_description_text)

    similarity_score = calculate_similarity(processed_resume_text, processed_job_description_text)
    keyword_analysis = analyze_keywords(processed_resume_text, processed_job_description_text)

    print(f"Similarity Score: {similarity_score:.2f}")
    print("\nKeyword Analysis:")
    print(f"  Resume Keywords: {keyword_analysis['resume_keywords'].most_common(10)}")
    print(f"  Job Description Keywords: {keyword_analysis['job_description_keywords'].most_common(10)}")
    print(f"  Common Keywords: {keyword_analysis['common_keywords']}")

    if similarity_score < 0.1:
        print("\nLow match. Focus on incorporating more relevant keywords and aligning your resume with the job description.")
    elif similarity_score < 0.3:
         print("\nFair match. Consider strengthening keyword usage and tailoring your resume further.")
    else:
        print("\nGood match. Your resume is well-aligned with the job description.")

if __name__ == "__main__":
    resume_file = input("Enter the path to your resume file (.txt or .pdf): ")
    job_description_file = input("Enter the path to the job description file (.txt or .pdf): ")
    ats_check(resume_file, job_description_file)