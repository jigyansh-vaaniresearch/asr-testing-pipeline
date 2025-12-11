import os
import glob
import csv
import re
import jiwer
from rapidfuzz import fuzz
from pathlib import Path

# --- CONFIGURATION ---
TRANSCRIPT_DIR = "raw_transcripts"
GROUND_TRUTH_DIR = "ground_truth2"
OUTPUT_CSV = "final_leaderboard_lean.csv"

def clean_text(text):
    """
    Strict cleaning: Removes metadata headers and standardizes spacing.
    """
    if not text: return ""
    # Remove metadata headers (Confidence: x, Transcript: y)
    text = re.sub(r'Confidence:.*', '', text) 
    text = re.sub(r'Transcript:.*', '', text)
    
    # Remove non-devanagari/english chars (keeps Hindi u0900-u097F + English + Numbers)
    text = re.sub(r'[^\u0900-\u097F\w\s]', '', text)
    
    # Collapse multiple spaces and lowercase
    return re.sub(r'\s+', ' ', text).strip().lower()

def parse_file(filepath):
    """Extracts Confidence and Transcript from the file"""
    if not os.path.exists(filepath):
        return 0.0, ""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Extract Confidence
    conf_match = re.search(r'Confidence:\s*([\d\.]+)', content)
    confidence = float(conf_match.group(1)) if conf_match else 0.0
    
    # 2. Extract Transcript (Everything after 'Transcript:')
    if "Transcript:" in content:
        transcript = content.split("Transcript:", 1)[1]
    else:
        # Fallback: Treat whole file as transcript if no header found
        # (Be careful, this might include the 'Confidence:' line if logic fails, 
        # but the regex above helps safely extract confidence first)
        transcript = re.sub(r'Confidence:\s*[\d\.]+', '', content)
        
    return confidence, transcript

def calculate_fuzzy_wer(ref, hyp, threshold=85):
    """
    Calculates WER where words with >85% similarity are considered correct.
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    if not ref_words: return 1.0
    if not hyp_words: return 1.0
    
    import difflib
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    matches = 0
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            matches += (i2 - i1)
        elif tag == 'replace':
            # Check fuzzy match for replaced words
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if fuzz.ratio(ref_words[i], hyp_words[j]) >= threshold:
                    matches += 1
                    
    # Fuzzy WER = 1 - (Matching Words / Total Reference Words)
    return max(0.0, 1.0 - (matches / len(ref_words)))

def main():
    print(f" Starting Lean Evaluation...")
    print(f" Output will be saved to: {OUTPUT_CSV}")
    
    if not os.path.exists(TRANSCRIPT_DIR):
        print(f" Error: Directory '{TRANSCRIPT_DIR}' not found.")
        return

    # Get all ground truth files
    gt_files = glob.glob(os.path.join(GROUND_TRUTH_DIR, "*.txt"))
    
    # Get all model folders
    models = [d for d in os.listdir(TRANSCRIPT_DIR) if os.path.isdir(os.path.join(TRANSCRIPT_DIR, d))]
    
    csv_data = []
    
    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        stem = Path(filename).stem
        
        # Read Ground Truth
        with open(gt_path, 'r', encoding='utf-8') as f:
            raw_gt = f.read()
        norm_gt = clean_text(raw_gt)
        
        for model in models:
            pred_path = os.path.join(TRANSCRIPT_DIR, model, filename)
            
            # Parse Prediction
            confidence, raw_hyp = parse_file(pred_path)
            norm_hyp = clean_text(raw_hyp)
            
            # Calculate Metrics
            if not norm_hyp:
                wer = 1.0
                fuzzy_wer = 1.0
            else:
                wer = jiwer.wer(norm_gt, norm_hyp)
                fuzzy_wer = calculate_fuzzy_wer(norm_gt, norm_hyp)
            
            # Append Row [Filename, Model, Confidence, WER, Fuzzy WER]
            csv_data.append([
                filename,
                model,
                confidence,
                round(wer, 4),
                round(fuzzy_wer, 4)
            ])
            
            print(f"   Processed {filename} | {model} | WER: {wer:.2f}")

    # Save to CSV
    headers = ["Filename", "Model", "Confidence", "WER", "Fuzzy WER"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_data)
        
    print(f"\n Done! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()