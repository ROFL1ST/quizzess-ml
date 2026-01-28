import sys
import os
import pandas as pd
import time

# Add parent directory to sys.path to import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.services.grading import grading_service

def run_experiment():
    print("🚀 Starting Experiment...")
    
    # 1. Load Dataset
    input_file = os.path.join(current_dir, "essay_dataset_v1.csv")
    if not os.path.exists(input_file):
        print(f"❌ Dataset not found: {input_file}")
        return
        
    df = pd.read_csv(input_file)
    print(f"📄 Loaded {len(df)} rows from dataset.")
    
    # 2. Run Grading
    tfidf_scores = []
    sbert_scores = []
    times = []
    
    print("🧠 Processing grades (SBERT + TFHUB)...")
    
    for index, row in df.iterrows():
        start_time = time.time()
        
        # Call Internal Service Directly
        result = grading_service.grade(row['teacher_key'], row['student_answer'])
        
        end_time = time.time()
        
        # Extract Scores from Debug Info
        debug = result.get('debug', {})
        tfidf_s = debug.get('score_tfidf', 0.0)
        sbert_s = debug.get('score_sbert', 0.0)
        
        tfidf_scores.append(tfidf_s)
        sbert_scores.append(sbert_s)
        times.append(end_time - start_time)
        
        # Print progess every 10 rows
        if (index + 1) % 5 == 0:
            print(f"   Processed {index + 1}/{len(df)}...")

    # 3. Save Results
    df['score_tfidf'] = tfidf_scores
    df['score_sbert'] = sbert_scores
    df['processing_time'] = times
    
    # Calculate Deviation (Error)
    df['error_tfidf'] = abs(df['label_human'] - df['score_tfidf'])
    df['error_sbert'] = abs(df['label_human'] - df['score_sbert'])
    
    output_file = os.path.join(current_dir, "experiment_results.csv")
    df.to_csv(output_file, index=False)
    
    print(f"✅ Experiment Completed!")
    print(f"📂 Results saved to: {output_file}")
    
    # 4. Quick Analysis
    mae_tfidf = df['error_tfidf'].mean()
    mae_sbert = df['error_sbert'].mean()
    
    corr_tfidf = df['label_human'].corr(df['score_tfidf'])
    corr_sbert = df['label_human'].corr(df['score_sbert'])
    
    print("\n📊 --- Short Report ---")
    print(f"MAE (Mean Absolute Error) TF-IDF: {mae_tfidf:.2f}")
    print(f"MAE (Mean Absolute Error) SBERT : {mae_sbert:.2f} (Lower is better)")
    print("-" * 30)
    print(f"Correlation (Pearson) TF-IDF : {corr_tfidf:.4f}")
    print(f"Correlation (Pearson) SBERT  : {corr_sbert:.4f} (Higher is better)")
    print("-" * 30)
    
    if mae_sbert < mae_tfidf:
        print("🏆 Conclusion: SBERT performs better!")
    else:
        print("🏆 Conclusion: TF-IDF performs better!")

if __name__ == "__main__":
    run_experiment()
