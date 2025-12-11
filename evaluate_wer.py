import os
import re
import glob
import time
import csv
from dotenv import load_dotenv
from openai import OpenAI
import jiwer
from pathlib import Path

load_dotenv()

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TextNormalizer:
    def __init__(self):
        self.client = client

    def clean_ground_truth(self, text):
        """
        Aggressively removes speaker labels and metadata.
        """
        # Remove standard labels
        text = re.sub(r'^(agent|customer|speaker \d+|unknown):\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        # Remove timestamps if any [00:00]
        text = re.sub(r'\[\d{2}:\d{2}\]', '', text)
        # Replace newlines with spaces
        text = text.replace('\n', ' ').strip()
        return text

    def clean_model_transcript(self, text, model_name):
        """
        Cleans up specific artifacts based on model or common patterns.
        """
        if "[AAI Final]:" in text:
            lines = text.split('\n')
            clean_lines = []
            for line in lines:
                if "[AAI Final]:" in line:
                    try:
                        parts = line.split("[AAI Final]:")
                        if len(parts) > 1:
                            content = parts[1].strip()
                            if content:
                                clean_lines.append(content)
                    except IndexError:
                        continue
            text = " ".join(clean_lines)
            
        match = re.search(r'Transcript:\s*(.*)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        return text.replace('\n', ' ').strip()

    def llm_normalize_script(self, text, label="text"):
        """
        Uses OpenAI GPT to convert mixed Hindi/English script into 
        standardized Romanized Hinglish.
        """
        if not text or len(text) < 2:
            return ""
            
        system_prompt = """
        You are a precise transliteration engine for Hindi-English code-switched text.
        Your ONLY task is to convert Devanagari script to standardized Romanized English characters (Hinglish).

        STANDARDIZATION RULES:
        1. **Standardize Spelling:**
           - "मैं" -> "main" (not "mai" or "me")
           - "है" -> "hai"
           - "नहीं" -> "nahi"
           - "हाँ" -> "haan"
           - "जी" -> "ji"
           - "कैसे" -> "kaise" (not "kese")
           - "पैसे" -> "paise"
        2. **Preserve English:** Keep existing English words exactly as they are.
        3. **Structure:** Do NOT add words. Do NOT fix grammar. Output raw lowercase text only.
        4. **Clean:** Remove all punctuation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0 
            )
            normalized = response.choices[0].message.content.strip().lower()
            normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
            return normalized
            
        except Exception as e:
            print(f"      ⚠️ LLM Normalization Failed (Using raw text): {e}")
            return text.lower()

def post_process_normalization(text):
    """
    Manual mapping for common phonetic mismatches to improve WER accuracy.
    This helps match 'kese' to 'kaise', 'mai' to 'main', etc.
    """
    replacements = {
        r'\bmai\b': 'main',
        r'\bme\b': 'main',
        r'\bkese\b': 'kaise',
        r'\bhaan\b': 'han',
        r'\bh\b': 'hai',
        r'\bnhi\b': 'nahi',
        r'\bnahi\b': 'nahi',
        r'\bokay\b': 'ok',
        r'\bya\b': 'yeah',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text

def calculate_wer(reference, hypothesis):
    # 1. Initial whitespace cleanup
    reference = re.sub(r'\s+', ' ', reference).strip()
    hypothesis = re.sub(r'\s+', ' ', hypothesis).strip()
    
    # 2. Apply Phonetic Post-Processing (Manual Mapping)
    reference = post_process_normalization(reference)
    hypothesis = post_process_normalization(hypothesis)
    
    if not reference:
        return 1.0, "Empty Reference"
    if not hypothesis:
        return 1.0, "Empty Hypothesis"
    
    try:
        wer = jiwer.wer(reference, hypothesis)
    except Exception as e:
        print(f"      ❌ Jiwer Error: {e}")
        return 1.0, "Error"
        
    return wer, {}

def main():
    # --- DIRECTORIES ---
    dirs = {
        "Ground Truth": "ground_truth",
        "Deepgram": "deepgram_transcript",
        "Gladia": "gladia_transcript",
        "Speechmatics": "speechmatics_transcript",
        "AssemblyAI": "assemblyai_transcript"
    }

    if not os.path.exists(dirs["Ground Truth"]):
        print(f"❌ Error: Directory '{dirs['Ground Truth']}' not found.")
        return

    print("🔄 Initializing Normalizer (OpenAI)...")
    normalizer = TextNormalizer()
    report_data = []
    
    gt_files = glob.glob(os.path.join(dirs["Ground Truth"], "*.txt"))
    
    if not gt_files:
        print(f"❌ No text files found inside '{dirs['Ground Truth']}'")
        return

    print(f"🚀 Starting End-to-End WER Evaluation on {len(gt_files)} files...")
    print("-" * 60)

    for gt_path in gt_files:
        filename = os.path.basename(gt_path)
        file_stem = Path(filename).stem
        
        print(f"\n📄 Processing: {filename}")

        # --- A. Process Ground Truth ---
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                raw_gt = f.read()
            
            clean_gt = normalizer.clean_ground_truth(raw_gt)
            normalized_gt = normalizer.llm_normalize_script(clean_gt, "Ground Truth")
            
            if not normalized_gt:
                print("   ⚠️ Skipping file: Normalized GT is empty.")
                continue
                
        except Exception as e:
            print(f"   ❌ Error reading GT: {e}")
            continue

        # --- B. Process Each Model ---
        for model_name, model_dir in dirs.items():
            if model_name == "Ground Truth": continue

            model_file_path = os.path.join(model_dir, filename)
            
            if not os.path.exists(model_file_path):
                alt_path = os.path.join(model_dir, f"{filename}.wav.txt") 
                alt_path_2 = os.path.join(model_dir, f"{file_stem}.txt")
                
                if os.path.exists(alt_path):
                    model_file_path = alt_path
                elif os.path.exists(alt_path_2):
                    model_file_path = alt_path_2
                else:
                    report_data.append([filename, model_name, "N/A", normalized_gt, "MISSING_FILE"])
                    continue

            try:
                with open(model_file_path, 'r', encoding='utf-8') as f:
                    raw_hyp = f.read()

                clean_hyp = normalizer.clean_model_transcript(raw_hyp, model_name)
                normalized_hyp = normalizer.llm_normalize_script(clean_hyp, model_name)

                wer_score, _ = calculate_wer(normalized_gt, normalized_hyp)
                
                print(f"   📊 {model_name}: {wer_score:.4f}")
                
                report_data.append([
                    filename,
                    model_name,
                    wer_score,
                    normalized_gt,
                    normalized_hyp
                ])
            except Exception as e:
                print(f"   ❌ Error processing {model_name}: {e}")

    # --- 3. Reports ---
    csv_filename = "final_wer_report.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Model", "WER", "Normalized Reference", "Normalized Hypothesis"])
        writer.writerows(report_data)
    
    print("\n" + "="*60)
    print(f"💾 Saved report to: {csv_filename}")
    print("="*60)

    # --- 4. Leaderboard ---
    model_stats = {}
    for row in report_data:
        model = row[1]
        wer = row[2]
        if wer == "N/A": continue
        
        if model not in model_stats:
            model_stats[model] = []
        model_stats[model].append(float(wer))

    print("\n🏆 AVERAGE WER LEADERBOARD (Lower is Better)")
    print("-" * 45)
    print(f"{'MODEL':<20} | {'AVG WER':<10} | {'SAMPLES':<10}")
    print("-" * 45)
    
    averages = []
    for model, wers in model_stats.items():
        if not wers: continue
        avg_wer = sum(wers) / len(wers)
        averages.append((model, avg_wer, len(wers)))
    
    averages.sort(key=lambda x: x[1])
    
    for model, avg, count in averages:
        print(f"{model:<20} | {avg:.4f}     | {count:<10}")
    print("="*60)

    # --- 5. Save Comparison for Inspection ---
    with open("final_wer_comparison.txt", "w", encoding="utf-8") as f:
        for row in report_data:
            f.write(f"File: {row[0]} | Model: {row[1]} | WER: {row[2]}\n")
            f.write(f"REF: {row[3]}\n")
            f.write(f"HYP: {row[4]}\n")
            f.write("-" * 50 + "\n")
    print(f"💾 Saved detailed comparison to: final_wer_comparison.txt")

if __name__ == "__main__":
    main()