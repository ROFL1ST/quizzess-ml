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
    indobert_scores = []
    times = []
    
    print("🧠 Processing grades (SBERT + IndoBERT + TF-IDF)...")
    
    for index, row in df.iterrows():
        start_time = time.time()
        
        # Call Internal Service Directly
        result = grading_service.grade(row['teacher_key'], row['student_answer'])
        
        end_time = time.time()
        
        # Extract Scores from Debug Info
        debug = result.get('debug', {})
        tfidf_s = debug.get('score_tfidf', 0.0)
        sbert_s = debug.get('score_sbert', 0.0)
        indobert_s = debug.get('score_indobert', 0.0)
        
        tfidf_scores.append(tfidf_s)
        sbert_scores.append(sbert_s)
        indobert_scores.append(indobert_s)
        times.append(end_time - start_time)
        
        # Print progess every 10 rows
        if (index + 1) % 5 == 0:
            print(f"   Processed {index + 1}/{len(df)}...")

    # 3. Save Results
    df['score_tfidf'] = tfidf_scores
    df['score_sbert'] = sbert_scores
    df['score_indobert'] = indobert_scores
    df['processing_time'] = times
    
    # Calculate Deviation (Error)
    df['error_tfidf'] = abs(df['label_human'] - df['score_tfidf'])
    df['error_sbert'] = abs(df['label_human'] - df['score_sbert'])
    df['error_indobert'] = abs(df['label_human'] - df['score_indobert'])
    
    # 4. Quick Analysis (Print BEFORE saving to ensure visibility)
    mae_tfidf = df['error_tfidf'].mean()
    mae_sbert = df['error_sbert'].mean()
    mae_indobert = df['error_indobert'].mean()
    
    corr_tfidf = df['label_human'].corr(df['score_tfidf'])
    corr_sbert = df['label_human'].corr(df['score_sbert'])
    corr_indobert = df['label_human'].corr(df['score_indobert'])
    
    print("\n📊 --- Research Report (Model Comparison) ---")
    print(f"{'Model':<15} | {'MAE (Lower Better)':<20} | {'Correlation (Higher Better)':<25}")
    print("-" * 65)
    print(f"{'TF-IDF':<15} | {mae_tfidf:<20.2f} | {corr_tfidf:<25.4f}")
    print(f"{'SBERT (En)':<15} | {mae_sbert:<20.2f} | {corr_sbert:<25.4f}")
    print(f"{'IndoBERT (Id)':<15} | {mae_indobert:<20.2f} | {corr_indobert:<25.4f}")
    print("-" * 65)
    
    best_mae = min(mae_tfidf, mae_sbert, mae_indobert)
    if mae_indobert == best_mae:
        print("🏆 Winner: IndoBERT (Most Accurate)")
    elif mae_sbert == best_mae:
        print("🏆 Winner: SBERT (Most Accurate)")
    else:
        print("🏆 Winner: TF-IDF (Most Accurate)")

    # 3. Save Results (Safe Save)
    output_file = os.path.join(current_dir, "experiment_results.csv")
    try:
        df.to_csv(output_file, index=False)
        print(f"\n✅ Experiment Completed!")
        print(f"📂 Results saved to: {output_file}")
    except PermissionError:
        print(f"\n⚠️ Could not save to {output_file} (File locked).")
        alt_file = os.path.join(current_dir, "experiment_results_new.csv")
        df.to_csv(alt_file, index=False)
        print(f"📂 Saved to alternate file: {alt_file}")

if __name__ == "__main__":
    run_experiment()
