import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
file_path = "data/jobs2.csv"
df = pd.read_csv(file_path)

# Define your skillset
my_skills = {"React", "Next.js", "Flutter", "Firebase", "SQL", "Java", "Python", "AWS", "DynamoDB", "S3", "Express", "Socket.io", "Prisma", "PostgreSQL"}

# Function to calculate score based on skill and description match
def calculate(df, resume_skills):
    resume_skills = ' '.join(resume_skills).lower().strip()
    
    df['skills'] = df['skills'].fillna('').str.lower().str.strip()
    df['description'] = df['description'].fillna('').str.lower().str.strip()
    
    scores = []
    for index, row in df.iterrows():
        job_skills = row['skills']
        job_description = row['description']

        # Debugging output
        print(f"\n[DEBUG] Job {index+1}:")
        print(f"Job Skills: {job_skills}")
        print(f"Job Description: {job_description}")
        print(f"Resume Skills: {resume_skills}")

        if not job_skills and not job_description:
            print("[DEBUG] Skipping due to missing data")
            scores.append(0)
            continue

        text_data = [resume_skills, job_skills + " " + job_description]
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(text_data)
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]

        print(f"[DEBUG] Similarity Score: {similarity}")
        scores.append(similarity)

    df['score'] = np.array(scores) * 100
    return df

# Apply scoring function
df = calculate(df, my_skills)

# Sort by score
df_sorted = df.sort_values(by='score', ascending=False)

# Save sorted results
df_sorted.to_csv("data/sorted_jobs.csv", index=False)

# Display top results
df_sorted[['title', 'company', 'location', 'score']].head(10)
