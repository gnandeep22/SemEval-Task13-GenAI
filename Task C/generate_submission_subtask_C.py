"""
Competition Submission Generator for Hybrid Code Detection
===========================================================
Generates predictions in CSV format required by scorer.py

Usage:
    python generate_submission.py --input task_c_test.parquet --output submission_subtask_c.csv
    python generate_submission.py --input task_c_validation.parquet --output validation_predictions.csv
"""

import pandas as pd
import numpy as np
import joblib
import re
import argparse
import os
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = r'C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_C\hybrid_code_model.pkl'

# Default input/output paths (UPDATED TO NEW NAMES)
DEFAULT_TEST_INPUT = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_C\task_c_test_set_sample.parquet"
DEFAULT_VALIDATION_INPUT = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_C\task_c_validation_set.parquet"
DEFAULT_OUTPUT = r'C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_C\submission_subtask_c.csv'
DEFAULT_GOLD_OUTPUT = r'C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_C\gold_subtask_c.csv'

LABEL_MAP = {
    0: 'human',
    1: 'machine',
    2: 'hybrid',
    3: 'adversarial'
}

# ============================================================================
# FEATURE EXTRACTION (Same as training)
# ============================================================================

def count_nesting(code: str) -> int:
    """Count maximum nesting level"""
    max_depth = 0
    current_depth = 0
    for char in code:
        if char in '{([':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char in '})]':
            current_depth = max(0, current_depth - 1)
    return max_depth


def extract_features(code: str) -> np.ndarray:
    """Extract 35 features from code snippet (must match training)"""
    if not isinstance(code, str) or not code:
        return np.zeros(35)
    
    lines = code.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    features = []
    
    # BASIC METRICS (4)
    features.append(len(lines))
    features.append(len(code))
    features.append(len(code) / max(len(lines), 1))
    features.append(max([len(l) for l in lines], default=0))
    
    # WHITESPACE (3)
    features.append(sum([1 for l in lines if l.startswith(' ')]) / max(len(lines), 1))
    features.append(1 if '\t' in code else 0)
    features.append(1 if re.search(r'\s+$', code, re.MULTILINE) else 0)
    
    # COMMENTS (3)
    comment_count = len(re.findall(r'//|/\*|#|"""', code))
    features.append(comment_count)
    features.append(comment_count / max(len(lines), 1))
    features.append(1 if re.search(r'Example:|Usage:|Returns:|Args:', code) else 0)
    
    # NAMING (4)
    features.append(len(re.findall(r'[a-z]+[A-Z]', code)))
    features.append(len(re.findall(r'[a-z]+_[a-z]+', code)))
    features.append(len(re.findall(r'\b[a-z]\b', code)))
    features.append(len(re.findall(r'[a-z]{8,}', code)))
    
    # TYPE HINTS (2)
    features.append(1 if re.search(r':\s*(int|str|float|bool|List|Dict|Optional|Any)', code) else 0)
    features.append(len(re.findall(r'""".*?"""', code, re.DOTALL)))
    
    # CODE STRUCTURE (3)
    features.append(len(re.findall(r'def |function |class |interface ', code)))
    features.append(1 if re.search(r'try|except|catch|finally', code) else 0)
    features.append(1 if re.search(r'print|console\.log|logger', code) else 0)
    
    # CONTROL FLOW (3)
    features.append(count_nesting(code))
    features.append(len(re.findall(r'if |while |for ', code)))
    features.append(len(re.findall(r'return ', code)))
    
    # INDENTATION (1)
    indents = [len(l) - len(l.lstrip()) for l in non_empty_lines if l.strip()]
    indent_diffs = [abs(indents[i] - indents[i-1]) for i in range(1, len(indents))]
    unique_diffs = len(set([d for d in indent_diffs if d > 0]))
    features.append(1 if unique_diffs <= 2 else 0)
    
    # COMMENT VARIANCE (1)
    comments = re.findall(r'//.*|#.*', code)
    if len(comments) > 1:
        lengths = [len(c) for c in comments]
        variance = np.var(lengths)
        features.append(min(variance / 100, 10))
    else:
        features.append(0)
    
    # MARKERS (1)
    features.append(1 if re.search(r'TODO|FIXME|HACK|XXX', code) else 0)
    
    # ADVANCED PATTERNS (5)
    features.append(len(re.findall(r'import |require\(|from .* import', code)))
    features.append(len(re.findall(r'===|!==|==|!=', code)))
    features.append(len(re.findall(r'\{|\}', code)))
    features.append(len(re.findall(r'\(|\)', code)))
    features.append(1 if re.search(r'async|await|Promise', code) else 0)
    
    # CODE QUALITY (5)
    features.append(len(re.findall(r'[A-Z]{2,}', code)))
    features.append(len(non_empty_lines) / max(len(lines), 1))
    features.append(1 if re.search(r'lambda |=>|\bmap\b|\bfilter\b', code) else 0)
    features.append(code.count(';') / max(len(lines), 1))
    features.append(1 if len(code) > 500 and comment_count == 0 else 0)
    
    return np.array(features)


# ============================================================================
# PREDICTION GENERATOR
# ============================================================================

def generate_predictions(input_file, output_file, model_path):
    """
    Generate predictions in competition format.
    
    Expected output CSV format:
    ID,prediction
    1,human
    2,machine
    3,hybrid
    ...
    """
    
    print("="*70)
    print("COMPETITION SUBMISSION GENERATOR")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n✗ Error: Model not found at {model_path}")
        print("Please train the model first!")
        return False
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"\n✗ Error: Input file not found at {input_file}")
        return False
    
    try:
        # Load model
        print(f"\n[1/4] Loading trained model...")
        model_data = joblib.load(model_path)
        model = model_data['model']
        label_map = model_data['label_map']
        print(f"   ✓ Model loaded successfully")
        print(f"   Training date: {model_data.get('training_date', 'Unknown')}")
        print(f"   Training accuracy: {model_data.get('accuracy', 0)*100:.2f}%")
        
        # Load input data
        print(f"\n[2/4] Loading input data...")
        print(f"   Path: {input_file}")
        df = pd.read_parquet(input_file)
        print(f"   ✓ Loaded {len(df):,} samples")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Check for required columns
        if 'code' not in df.columns:
            print("\n✗ Error: Input file must contain 'code' column")
            return False
        
        # Check for ID column (required for submission)
        if 'ID' not in df.columns and 'id' not in df.columns:
            print("\n⚠ Warning: No ID column found. Creating IDs from index...")
            df['ID'] = range(len(df))
        elif 'id' in df.columns and 'ID' not in df.columns:
            df['ID'] = df['id']
        
        # Extract features
        print(f"\n[3/4] Extracting features and generating predictions...")
        tqdm.pandas(desc="   Processing")
        
        features_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Extracting features"):
            features = extract_features(row['code'])
            features_list.append(features)
        
        X = np.array(features_list)
        print(f"   ✓ Feature extraction complete")
        
        # Make predictions
        print(f"   Making predictions...")
        predictions_numeric = model.predict(X)
        
        # Create submission dataframe with TEXT predictions
        predictions_text = [label_map[pred] for pred in predictions_numeric]
        submission = pd.DataFrame({
            'ID': df['ID'].values,
            'prediction': predictions_text  # Convert to text: human, machine, hybrid, adversarial
        })
        
        # Save to CSV
        print(f"\n[4/4] Saving predictions...")
        submission.to_csv(output_file, index=False)
        print(f"   ✓ Predictions saved to: {output_file}")
        
        # Show sample predictions
        print(f"\n   Sample predictions (first 10):")
        print(submission.head(10).to_string(index=False))
        
        # Show prediction distribution
        print(f"\n   Prediction distribution:")
        for label_name, count in submission['prediction'].value_counts().items():
            percentage = (count / len(submission)) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {label_name:12} {count:6,} ({percentage:5.2f}%) {bar}")
        
        print("\n" + "="*70)
        print("✓ SUBMISSION FILE CREATED SUCCESSFULLY!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# CREATE GOLD LABELS FILE (For testing scorer)
# ============================================================================

def create_gold_labels(input_file, output_file):
    """
    Create gold labels CSV from parquet file (for testing with scorer.py)
    
    Output format:
    ID,label
    1,human
    2,machine
    3,hybrid
    ...
    """
    print("="*70)
    print("GOLD LABELS GENERATOR")
    print("="*70)
    
    if not os.path.exists(input_file):
        print(f"\n✗ Error: Input file not found at {input_file}")
        return False
    
    try:
        print(f"\n[1/2] Loading data...")
        df = pd.read_parquet(input_file)
        print(f"   ✓ Loaded {len(df):,} samples")
        
        if 'label' not in df.columns:
            print("\n✗ Error: Input file must contain 'label' column")
            return False
        
        # Check for ID column
        if 'ID' not in df.columns and 'id' not in df.columns:
            print("\n⚠ Warning: No ID column found. Creating IDs from index...")
            df['ID'] = range(len(df))
        elif 'id' in df.columns and 'ID' not in df.columns:
            df['ID'] = df['id']
        
        print(f"\n[2/2] Creating gold labels file...")
        
        # Convert numeric labels to text
        labels_text = [LABEL_MAP[label] for label in df['label'].values]
        gold_labels = pd.DataFrame({
            'ID': df['ID'].values,
            'label': labels_text
        })
        
        gold_labels.to_csv(output_file, index=False)
        print(f"   ✓ Gold labels saved to: {output_file}")
        
        print(f"\n   Sample gold labels (first 10):")
        print(gold_labels.head(10).to_string(index=False))
        
        print("\n" + "="*70)
        print("✓ GOLD LABELS FILE CREATED!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate competition submission file for Hybrid Code Detection"
    )
    
    parser.add_argument(
        '--input',
        default=DEFAULT_TEST_INPUT,
        help='Path to input parquet file (e.g., task_c_test.parquet)'
    )
    
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help='Path to output CSV file (default: submission_subtask_c.csv)'
    )
    
    parser.add_argument(
        '--model',
        default=MODEL_PATH,
        help=f'Path to trained model (default: {MODEL_PATH})'
    )
    
    parser.add_argument(
        '--gold-input',
        default=DEFAULT_VALIDATION_INPUT,
        help='Path to validation parquet file for gold labels'
    )
    
    parser.add_argument(
        '--gold-output',
        default=DEFAULT_GOLD_OUTPUT,
        help='Path to gold labels CSV file (default: gold_subtask_c.csv)'
    )
    
    parser.add_argument(
        '--skip-gold',
        action='store_true',
        help='Skip creating gold labels file'
    )
    
    args = parser.parse_args()
    
    # Generate predictions
    print("\n" + "="*70)
    print("STEP 1: GENERATING PREDICTIONS")
    print("="*70)
    generate_predictions(args.input, args.output, args.model)
    
    # Generate gold labels (unless skipped)
    if not args.skip_gold:
        print("\n\n" + "="*70)
        print("STEP 2: GENERATING GOLD LABELS")
        print("="*70)
        create_gold_labels(args.gold_input, args.gold_output)
    
    print("\n\n" + "="*70)
    print("ALL FILES GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {args.output}")
    if not args.skip_gold:
        print(f"  2. {args.gold_output}")


if __name__ == "__main__":
    main()