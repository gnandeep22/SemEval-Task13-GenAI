"""
Subtask C: Hybrid Code Detection
=================================
A machine learning tool to classify code snippets as:
- Human-written
- Machine-generated
- Hybrid (human + machine)
- Adversarial (deliberately obfuscated)

Usage:
    Training:   python train_subtask_c.py --train
    Testing:    python train_subtask_c.py --test
    Validation: python train_subtask_c.py --validate
    Prediction: python train_subtask_c.py --predict
    Auto:       python train_subtask_c.py (trains if no model exists, then predicts)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import re
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Local file paths - UPDATE THESE TO YOUR ACTUAL PATHS
INPUT_FILE = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_C\task_c_training_set_1.parquet"
TEST_FILE = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_C\task_c_test_set_sample.parquet"
VALIDATION_FILE = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_C\task_c_validation_set.parquet"
MODEL_OUTPUT = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_C\hybrid_code_model.pkl"

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 20

# Label definitions
LABEL_MAP = {
    0: 'human',
    1: 'machine',
    2: 'hybrid',
    3: 'adversarial'
}

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(code: str) -> np.ndarray:
    """Extract 35 statistical and structural features from code snippet."""
    if not isinstance(code, str) or not code:
        return np.zeros(35)
    
    lines = code.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    
    features = []
    
    # BASIC METRICS (4 features)
    features.append(len(lines))
    features.append(len(code))
    features.append(len(code) / max(len(lines), 1))
    features.append(max([len(l) for l in lines], default=0))
    
    # WHITESPACE PATTERNS (3 features)
    features.append(sum([1 for l in lines if l.startswith(' ')]) / max(len(lines), 1))
    features.append(1 if '\t' in code else 0)
    features.append(1 if re.search(r'\s+$', code, re.MULTILINE) else 0)
    
    # COMMENTS (3 features)
    comment_count = len(re.findall(r'//|/\*|#|"""', code))
    features.append(comment_count)
    features.append(comment_count / max(len(lines), 1))
    features.append(1 if re.search(r'Example:|Usage:|Returns:|Args:', code) else 0)
    
    # NAMING CONVENTIONS (4 features)
    features.append(len(re.findall(r'[a-z]+[A-Z]', code)))
    features.append(len(re.findall(r'[a-z]+_[a-z]+', code)))
    features.append(len(re.findall(r'\b[a-z]\b', code)))
    features.append(len(re.findall(r'[a-z]{8,}', code)))
    
    # TYPE HINTS & DOCUMENTATION (2 features)
    features.append(1 if re.search(r':\s*(int|str|float|bool|List|Dict|Optional|Any)', code) else 0)
    features.append(len(re.findall(r'""".*?"""', code, re.DOTALL)))
    
    # CODE STRUCTURE (3 features)
    features.append(len(re.findall(r'def |function |class |interface ', code)))
    features.append(1 if re.search(r'try|except|catch|finally', code) else 0)
    features.append(1 if re.search(r'print|console\.log|logger', code) else 0)
    
    # CONTROL FLOW (3 features)
    features.append(count_nesting(code))
    features.append(len(re.findall(r'if |while |for ', code)))
    features.append(len(re.findall(r'return ', code)))
    
    # INDENTATION CONSISTENCY (1 feature)
    indents = [len(l) - len(l.lstrip()) for l in non_empty_lines if l.strip()]
    indent_diffs = [abs(indents[i] - indents[i-1]) for i in range(1, len(indents))]
    unique_diffs = len(set([d for d in indent_diffs if d > 0]))
    features.append(1 if unique_diffs <= 2 else 0)
    
    # COMMENT VARIANCE (1 feature)
    comments = re.findall(r'//.*|#.*', code)
    if len(comments) > 1:
        lengths = [len(c) for c in comments]
        variance = np.var(lengths)
        features.append(min(variance / 100, 10))
    else:
        features.append(0)
    
    # CODE MARKERS (1 feature)
    features.append(1 if re.search(r'TODO|FIXME|HACK|XXX', code) else 0)
    
    # ADVANCED PATTERNS (5 features)
    features.append(len(re.findall(r'import |require\(|from .* import', code)))
    features.append(len(re.findall(r'===|!==|==|!=', code)))
    features.append(len(re.findall(r'\{|\}', code)))
    features.append(len(re.findall(r'\(|\)', code)))
    features.append(1 if re.search(r'async|await|Promise', code) else 0)
    
    # CODE QUALITY INDICATORS (5 features)
    features.append(len(re.findall(r'[A-Z]{2,}', code)))
    features.append(len(non_empty_lines) / max(len(lines), 1))
    features.append(1 if re.search(r'lambda |=>|\bmap\b|\bfilter\b', code) else 0)
    features.append(code.count(';') / max(len(lines), 1))
    features.append(1 if len(code) > 500 and comment_count == 0 else 0)
    
    return np.array(features)


def count_nesting(code: str) -> int:
    """Calculate maximum nesting level in code by counting braces/brackets."""
    max_depth = 0
    current_depth = 0
    
    for char in code:
        if char in '{([':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char in '})]':
            current_depth = max(0, current_depth - 1)
    
    return max_depth


# ============================================================================
# EVALUATION HELPER
# ============================================================================

def evaluate_model(model, X, y, dataset_name="Dataset"):
    """Evaluate model and print detailed metrics."""
    print(f"\n{'='*70}")
    print(f"EVALUATING ON {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Make predictions
    print(f"\nMaking predictions on {len(X):,} samples...")
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_micro = f1_score(y, y_pred, average='micro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    f1_per_class = f1_score(y, y_pred, average=None)
    
    # Display metrics
    print(f"\n{'='*58}")
    print(f"{'PERFORMANCE METRICS':^58}")
    print(f"{'='*58}")
    print(f"Overall Accuracy:        {accuracy*100:6.2f}%")
    print(f"F1 Score (Macro):        {f1_macro*100:6.2f}%")
    print(f"F1 Score (Weighted):     {f1_weighted*100:6.2f}%")
    print(f"F1 Score (Micro):        {f1_micro*100:6.2f}%")
    print(f"{'='*58}")
    
    # F1 scores per class
    print(f"\nF1 Score by Class:")
    for i, f1 in enumerate(f1_per_class):
        bar = "█" * int(f1 * 50)
        print(f"  {LABEL_MAP[i]:12} {f1*100:6.2f}% {bar}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        y, y_pred, 
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
        digits=4
    ))
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    
    # Header
    print(f"  {'Actual →':12} " + " ".join([f"{LABEL_MAP[i]:>12}" for i in sorted(LABEL_MAP.keys())]))
    print(f"  {'Predicted ↓':12}" + "-" * (13 * len(LABEL_MAP)))
    
    # Rows
    for i, row in enumerate(cm):
        print(f"  {LABEL_MAP[i]:12} " + " ".join([f"{val:>12}" for val in row]))
    
    print(f"\n{'='*70}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_micro': f1_micro,
        'f1_per_class': {LABEL_MAP[i]: f1_per_class[i] for i in range(len(f1_per_class))}
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_model() -> bool:
    """Train Random Forest classifier on labeled code dataset."""
    print("=" * 70)
    print("TRAINING MODE - HYBRID CODE DETECTION (SUBTASK C)")
    print("=" * 70)
    
    try:
        print(f"\n[1/4] Loading training data...")
        
        # Check if file exists
        if not os.path.exists(INPUT_FILE):
            print(f"\n✗ Error: Training file not found: {INPUT_FILE}")
            return False
        
        print(f"   Path: {INPUT_FILE}")
        df = pd.read_parquet(INPUT_FILE)
        print(f"   ✓ Loaded {len(df):,} samples")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Validate columns
        if 'code' not in df.columns or 'label' not in df.columns:
            print("\n✗ Error: Dataset must contain 'code' and 'label' columns")
            return False
        
        # Map labels
        df['label_text'] = df['label'].map(LABEL_MAP)
        
        print(f"\n   Label distribution:")
        label_counts = df['label_text'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {label:12} {count:6,} ({percentage:5.2f}%) {bar}")
        
        # Extract features
        print(f"\n[2/4] Extracting features from code...")
        tqdm.pandas(desc="   Processing")
        X_train = np.array(df['code'].progress_apply(extract_features).tolist())
        y_train = df['label'].values
        
        print(f"   ✓ Feature matrix shape: {X_train.shape}")
        print(f"   ✓ Training samples: {len(X_train):,}")
        
        # Train model
        print(f"\n[3/4] Training Random Forest classifier...")
        print(f"   Parameters: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
        
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        model.fit(X_train, y_train)
        print(f"   ✓ Model trained!")
        
        # Calculate training metrics (for reference only)
        print(f"\n[4/4] Calculating training metrics...")
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        f1_macro = f1_score(y_train, y_pred, average='macro')
        f1_micro = f1_score(y_train, y_pred, average='micro')
        f1_weighted = f1_score(y_train, y_pred, average='weighted')
        f1_per_class = f1_score(y_train, y_pred, average=None)
        
        print(f"\n   {'='*58}")
        print(f"   {'TRAINING SET METRICS':^58}")
        print(f"   {'='*58}")
        print(f"   Overall Accuracy:        {accuracy*100:6.2f}%")
        print(f"   F1 Score (Macro):        {f1_macro*100:6.2f}%")
        print(f"   F1 Score (Weighted):     {f1_weighted*100:6.2f}%")
        print(f"   F1 Score (Micro):        {f1_micro*100:6.2f}%")
        print(f"   {'='*58}")
        
        print(f"\n   F1 Score by Class:")
        for i, f1 in enumerate(f1_per_class):
            bar = "█" * int(f1 * 50)
            print(f"   {LABEL_MAP[i]:12} {f1*100:6.2f}% {bar}")
        
        # Save model
        print(f"\n   Saving model to: {MODEL_OUTPUT}")
        os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
        
        feature_names = [
            "Total lines", "Total chars", "Avg chars/line", "Max line length",
            "Indented lines ratio", "Uses tabs", "Trailing whitespace",
            "Comment count", "Comment density", "Has structured docs",
            "camelCase count", "snake_case count", "Single letter vars", "Long var names",
            "Has type hints", "Docstring count",
            "Function/class defs", "Has error handling", "Has logging",
            "Max nesting", "Control statements", "Return statements",
            "Consistent indent", "Comment variance", "Has TODO markers",
            "Import statements", "Equality checks", "Brace count", "Paren count", "Has async",
            "Constants/acronyms", "Code density", "Functional programming", "Semicolons/line", "Large uncommented"
        ]
        
        joblib.dump({
            'model': model,
            'label_map': LABEL_MAP,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro,
            'f1_per_class': {LABEL_MAP[i]: f1_per_class[i] for i in range(len(f1_per_class))},
            'feature_names': feature_names,
            'training_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': len(df),
            'subtask': 'C'
        }, MODEL_OUTPUT)
        print(f"   ✓ Model saved!")
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TESTING
# ============================================================================

def test_model() -> bool:
    """Test the trained model on test dataset."""
    print("=" * 70)
    print("TESTING MODE - HYBRID CODE DETECTION (SUBTASK C)")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists(MODEL_OUTPUT):
        print(f"\n✗ Error: Model file not found: {MODEL_OUTPUT}")
        print("Please train the model first: python train_subtask_c.py --train")
        return False
    
    # Check if test file exists
    if not os.path.exists(TEST_FILE):
        print(f"\n✗ Error: Test file not found: {TEST_FILE}")
        print(f"Please ensure the file exists at: {TEST_FILE}")
        return False
    
    try:
        # Load model
        print(f"\n[1/3] Loading trained model...")
        model_data = joblib.load(MODEL_OUTPUT)
        model = model_data['model']
        print(f"   ✓ Model loaded!")
        print(f"   Training date: {model_data.get('training_date', 'Unknown')}")
        print(f"   Training accuracy: {model_data.get('accuracy', 0)*100:.2f}%")
        
        # Load test data
        print(f"\n[2/3] Loading test data...")
        print(f"   Path: {TEST_FILE}")
        df_test = pd.read_parquet(TEST_FILE)
        print(f"   ✓ Loaded {len(df_test):,} test samples")
        
        # Validate columns
        if 'code' not in df_test.columns or 'label' not in df_test.columns:
            print("\n✗ Error: Test dataset must contain 'code' and 'label' columns")
            return False
        
        # Extract features
        print(f"\n[3/3] Extracting features and evaluating...")
        tqdm.pandas(desc="   Processing")
        X_test = np.array(df_test['code'].progress_apply(extract_features).tolist())
        y_test = df_test['label'].values
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test, "TEST SET")
        
        print("\n" + "=" * 70)
        print("✓ TESTING COMPLETE!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# VALIDATION
# ============================================================================

def validate_model() -> bool:
    """Validate the trained model on validation dataset."""
    print("=" * 70)
    print("VALIDATION MODE - HYBRID CODE DETECTION (SUBTASK C)")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists(MODEL_OUTPUT):
        print(f"\n✗ Error: Model file not found: {MODEL_OUTPUT}")
        print("Please train the model first: python train_subtask_c.py --train")
        return False
    
    # Check if validation file exists
    if not os.path.exists(VALIDATION_FILE):
        print(f"\n✗ Error: Validation file not found: {VALIDATION_FILE}")
        print(f"Please ensure the file exists at: {VALIDATION_FILE}")
        return False
    
    try:
        # Load model
        print(f"\n[1/3] Loading trained model...")
        model_data = joblib.load(MODEL_OUTPUT)
        model = model_data['model']
        print(f"   ✓ Model loaded!")
        print(f"   Training date: {model_data.get('training_date', 'Unknown')}")
        print(f"   Training accuracy: {model_data.get('accuracy', 0)*100:.2f}%")
        
        # Load validation data
        print(f"\n[2/3] Loading validation data...")
        print(f"   Path: {VALIDATION_FILE}")
        df_val = pd.read_parquet(VALIDATION_FILE)
        print(f"   ✓ Loaded {len(df_val):,} validation samples")
        
        # Validate columns
        if 'code' not in df_val.columns or 'label' not in df_val.columns:
            print("\n✗ Error: Validation dataset must contain 'code' and 'label' columns")
            return False
        
        # Extract features
        print(f"\n[3/3] Extracting features and evaluating...")
        tqdm.pandas(desc="   Processing")
        X_val = np.array(df_val['code'].progress_apply(extract_features).tolist())
        y_val = df_val['label'].values
        
        # Evaluate
        results = evaluate_model(model, X_val, y_val, "VALIDATION SET")
        
        print("\n" + "=" * 70)
        print("✓ VALIDATION COMPLETE!")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# PREDICTION
# ============================================================================

def classify_code(code: str) -> Optional[Tuple[str, np.ndarray]]:
    """Classify a code snippet using trained model."""
    if not os.path.exists(MODEL_OUTPUT):
        print(f"\n✗ Error: Model file not found: {MODEL_OUTPUT}")
        print("Please run training mode first: python train_subtask_c.py --train")
        return None
    
    try:
        model_data = joblib.load(MODEL_OUTPUT)
        model = model_data['model']
        label_map = model_data['label_map']
        
        features = extract_features(code).reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        class_name = label_map[prediction]
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"\nPredicted Class: {class_name.upper()}")
        print(f"Confidence: {probabilities[prediction]*100:.2f}%")
        
        print(f"\nAll Class Probabilities:")
        for label_idx, prob in enumerate(probabilities):
            label_name = label_map[label_idx]
            bar = "█" * int(prob * 50)
            print(f"  {label_name:12} {prob*100:6.2f}% {bar}")
        
        print(f"\nModel Information:")
        if 'training_date' in model_data:
            print(f"  Trained: {model_data['training_date']}")
        if 'total_samples' in model_data:
            print(f"  Training samples: {model_data['total_samples']:,}")
        if 'accuracy' in model_data:
            print(f"  Accuracy: {model_data['accuracy']*100:.2f}%")
        
        print("=" * 60 + "\n")
        
        return class_name, probabilities
        
    except Exception as e:
        print(f"\n✗ Error during classification: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def predict_mode():
    """Interactive prediction mode for classifying code snippets."""
    print("=" * 70)
    print("PREDICTION MODE - HYBRID CODE DETECTION (SUBTASK C)")
    print("=" * 70)
    
    test_code = """
def calculate_sum(numbers: List[int]) -> int:
    '''Calculate sum of numbers'''
    return sum(numbers)
"""
    
    print("\nTesting with sample code...")
    print("Code snippet:")
    print("-" * 60)
    print(test_code)
    print("-" * 60)
    
    classify_code(test_code)
    
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nInstructions:")
    print("  1. Paste your code snippet")
    print("  2. Press Enter twice (empty line) to classify")
    print("  3. Type 'quit' to exit")
    print("=" * 70)
    
    while True:
        print("\n" + "-" * 70)
        print("Enter code to classify:")
        print("-" * 70)
        
        lines = []
        empty_line_count = 0
        
        while True:
            try:
                line = input()
                
                if line.lower() == 'quit':
                    print("\n✓ Exiting prediction mode...")
                    return
                
                if line == '':
                    empty_line_count += 1
                    if empty_line_count >= 2 and len(lines) > 0:
                        break
                else:
                    empty_line_count = 0
                
                lines.append(line)
                
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n\n✓ Exiting prediction mode...")
                return
        
        if lines:
            while lines and lines[-1] == '':
                lines.pop()
            
            if lines:
                user_code = '\n'.join(lines)
                classify_code(user_code)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for the application."""
    print("\n" + "=" * 70)
    print("HYBRID CODE DETECTOR - SUBTASK C")
    print("=" * 70)
    print(f"Version: 1.0")
    print(f"Model: Random Forest Classifier")
    print(f"Classes: {', '.join(LABEL_MAP.values())}")
    print("=" * 70)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--train':
            train_model()
        elif sys.argv[1] == '--test':
            test_model()
        elif sys.argv[1] == '--validate':
            validate_model()
        elif sys.argv[1] == '--predict':
            predict_mode()
        elif sys.argv[1] == '--help':
            print("\nUsage:")
            print("  Training:    python train_subtask_c.py --train")
            print("  Testing:     python train_subtask_c.py --test")
            print("  Validation:  python train_subtask_c.py --validate")
            print("  Prediction:  python train_subtask_c.py --predict")
            print("  Auto mode:   python train_subtask_c.py")
            print("  Help:        python train_subtask_c.py --help")
            print("\nFiles:")
            print(f"  Training:    {INPUT_FILE}")
            print(f"  Test:        {TEST_FILE}")
            print(f"  Validation:  {VALIDATION_FILE}")
            print(f"  Model:       {MODEL_OUTPUT}")
            print("\nAuto mode trains the model if not found, then starts prediction.")
        else:
            print(f"\n✗ Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Auto mode
        if os.path.exists(MODEL_OUTPUT):
            print("\n✓ Model found! Starting prediction mode...")
            print("(Use --train to retrain, --test to test, --validate to validate)")
            predict_mode()
        else:
            print("\n! No trained model found.")
            print("\nStarting training mode...")
            if train_model():
                print("\n✓ Training complete! Now starting prediction mode...")
                predict_mode()


if __name__ == "__main__":
    main()