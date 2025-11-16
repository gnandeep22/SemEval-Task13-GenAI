"""
Subtask A: Binary Code Detection - Submission Generator
========================================================
Generates predictions in CSV format

Usage:
    python generate_submission_subtask_a.py
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

MODEL_PATH = r'C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_A\binary_code_model.pkl'

# Default input/output paths
DEFAULT_TEST_INPUT = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_A\task_a_test_set_sample.parquet"
DEFAULT_VALIDATION_INPUT = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_A\task_a_validation_set.parquet"
DEFAULT_OUTPUT = r'C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_A\submission_task_a.csv'
DEFAULT_GOLD_OUTPUT = r'C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_A\gold_task_a.csv'

LABEL_MAP = {
    0: 'human',
    1: 'machine'
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
    """Extract features from code snippet (must match training)"""
    if not isinstance(code, str) or not code:
        return np.zeros(40)
    
    lines = code.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    features = []
    
    # BASIC METRICS (5)
    features.append(len(lines))
    features.append(len(code))
    features.append(len(code) / max(len(lines), 1))
    features.append(max([len(l) for l in lines], default=0))
    features.append(len(non_empty_lines) / max(len(lines), 1))
    
    # WHITESPACE & FORMATTING (4)
    features.append(sum([1 for l in lines if l.startswith(' ')]) / max(len(lines), 1))
    features.append(1 if '\t' in code else 0)
    features.append(1 if re.search(r'\s+$', code, re.MULTILINE) else 0)
    features.append(code.count(' ') / max(len(code), 1))
    
    # COMMENTS (5)
    comment_count = len(re.findall(r'//|/\*|#|"""', code))
    features.append(comment_count)
    features.append(comment_count / max(len(lines), 1))
    features.append(1 if re.search(r'Example:|Usage:|Returns:|Args:|Parameters:', code) else 0)
    docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))
    features.append(docstring_count)
    features.append(len(re.findall(r'//.*TODO|#.*TODO|FIXME|HACK|XXX', code)))
    
    # NAMING CONVENTIONS (5)
    features.append(len(re.findall(r'[a-z]+[A-Z]', code)))
    features.append(len(re.findall(r'[a-z]+_[a-z]+', code)))
    features.append(len(re.findall(r'\b[a-z]\b', code)))
    features.append(len(re.findall(r'[a-z]{10,}', code)))
    features.append(len(re.findall(r'\b[A-Z][a-z]+[A-Z]', code)))
    
    # TYPE HINTS & DOCUMENTATION (3)
    features.append(1 if re.search(r':\s*(int|str|float|bool|List|Dict|Optional|Any|void)', code) else 0)
    features.append(len(re.findall(r'->', code)))
    features.append(len(re.findall(r'@\w+', code)))
    
    # CODE STRUCTURE (5)
    features.append(len(re.findall(r'def |function |class |interface |struct ', code)))
    features.append(1 if re.search(r'try|except|catch|finally|throw', code) else 0)
    features.append(1 if re.search(r'print|console\.log|logger|cout|System\.out', code) else 0)
    features.append(len(re.findall(r'import |require\(|from .* import|#include|using ', code)))
    features.append(len(re.findall(r'return ', code)))
    
    # CONTROL FLOW (4)
    features.append(count_nesting(code))
    features.append(len(re.findall(r'\bif\b|\bwhile\b|\bfor\b', code)))
    features.append(len(re.findall(r'\belse\b|\belif\b', code)))
    features.append(len(re.findall(r'break|continue', code)))
    
    # INDENTATION CONSISTENCY (2)
    indents = [len(l) - len(l.lstrip()) for l in non_empty_lines if l.strip()]
    if len(indents) > 1:
        indent_diffs = [abs(indents[i] - indents[i-1]) for i in range(1, len(indents))]
        unique_diffs = len(set([d for d in indent_diffs if d > 0]))
        features.append(1 if unique_diffs <= 2 else 0)
        features.append(np.std(indents) if len(indents) > 1 else 0)
    else:
        features.append(0)
        features.append(0)
    
    # OPERATORS & SYMBOLS (4)
    features.append(len(re.findall(r'===|!==|==|!=|<=|>=', code)))
    features.append(len(re.findall(r'\{|\}', code)))
    features.append(len(re.findall(r'\(|\)', code)))
    features.append(code.count(';') / max(len(lines), 1))
    
    # CODE QUALITY INDICATORS (3)
    features.append(len(re.findall(r'[A-Z]{2,}', code)))
    features.append(1 if re.search(r'lambda |=>|\bmap\b|\bfilter\b|\breduce\b', code) else 0)
    features.append(1 if len(code) > 500 and comment_count == 0 else 0)
    
    return np.array(features)


# ============================================================================
# PREDICTION GENERATOR
# ============================================================================

def generate_predictions(input_file, output_file, model_path):
    """Generate predictions in competition format"""
    
    print("="*70)
    print("SUBTASK A: BINARY CODE DETECTION - SUBMISSION GENERATOR")
    print("="*70)
    
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first!")
        return False
    
    if not os.path.exists(input_file):
        print(f"\nError: Input file not found at {input_file}")
        return False
    
    try:
        # Load model
        print(f"\n[1/4] Loading trained model...")
        model_data = joblib.load(model_path)
        model = model_data['model']
        label_map = model_data['label_map']
        print(f"   Model loaded successfully")
        print(f"   Training date: {model_data.get('training_date', 'Unknown')}")
        print(f"   Training Macro F1: {model_data.get('macro_f1', 0):.4f}")
        
        # Load input data
        print(f"\n[2/4] Loading input data...")
        print(f"   Path: {input_file}")
        df = pd.read_parquet(input_file)
        print(f"   Loaded {len(df):,} samples")
        
        if 'code' not in df.columns:
            print("\nError: Input file must contain 'code' column")
            return False
        
        # Handle ID column
        if 'ID' not in df.columns and 'id' not in df.columns:
            print("\n   Warning: No ID column found. Creating IDs from index...")
            df['ID'] = range(len(df))
        elif 'id' in df.columns and 'ID' not in df.columns:
            df['ID'] = df['id']
        
        # Extract features
        print(f"\n[3/4] Extracting features and generating predictions...")
        
        features_list = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Extracting features"):
            features = extract_features(row['code'])
            features_list.append(features)
        
        X = np.array(features_list)
        print(f"   Feature extraction complete")
        
        # Make predictions
        print(f"   Making predictions...")
        predictions_numeric = model.predict(X)
        
        # Convert to text labels
        predictions_text = [label_map[pred] for pred in predictions_numeric]
        submission = pd.DataFrame({
            'ID': df['ID'].values,
            'prediction': predictions_text
        })
        
        # Save to CSV
        print(f"\n[4/4] Saving predictions...")
        submission.to_csv(output_file, index=False)
        print(f"   Predictions saved to: {output_file}")
        
        # Show sample predictions
        print(f"\n   Sample predictions (first 10):")
        print(submission.head(10).to_string(index=False))
        
        # Show prediction distribution
        print(f"\n   Prediction distribution:")
        for label_name, count in submission['prediction'].value_counts().items():
            percentage = (count / len(submission)) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {label_name:10} {count:6,} ({percentage:5.2f}%) {bar}")
        
        print("\n" + "="*70)
        print("SUBMISSION FILE CREATED SUCCESSFULLY!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# CREATE GOLD LABELS FILE
# ============================================================================

def create_gold_labels(input_file, output_file):
    """Create gold labels CSV from parquet file"""
    
    print("="*70)
    print("SUBTASK A: GOLD LABELS GENERATOR")
    print("="*70)
    
    if not os.path.exists(input_file):
        print(f"\nError: Input file not found at {input_file}")
        return False
    
    try:
        print(f"\n[1/2] Loading data...")
        df = pd.read_parquet(input_file)
        print(f"   Loaded {len(df):,} samples")
        
        if 'label' not in df.columns:
            print("\nError: Input file must contain 'label' column")
            return False
        
        # Handle ID column
        if 'ID' not in df.columns and 'id' not in df.columns:
            print("\n   Warning: No ID column found. Creating IDs from index...")
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
        print(f"   Gold labels saved to: {output_file}")
        
        print(f"\n   Sample gold labels (first 10):")
        print(gold_labels.head(10).to_string(index=False))
        
        # Show label distribution
        print(f"\n   Label distribution:")
        for label_name, count in gold_labels['label'].value_counts().items():
            percentage = (count / len(gold_labels)) * 100
            print(f"   {label_name:10} {count:6,} ({percentage:5.2f}%)")
        
        print("\n" + "="*70)
        print("GOLD LABELS FILE CREATED!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate submission file for Subtask A: Binary Code Detection"
    )
    
    parser.add_argument(
        '--input',
        default=DEFAULT_TEST_INPUT,
        help='Path to input parquet file (e.g., task_a_test.parquet)'
    )
    
    parser.add_argument(
        '--output',
        default=DEFAULT_OUTPUT,
        help='Path to output CSV file (e.g., submission_task_a.csv)'
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
        help='Path to gold labels CSV file'
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


if __name__ == "__main__":
    main()