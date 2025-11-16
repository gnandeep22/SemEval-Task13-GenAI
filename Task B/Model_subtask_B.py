"""
Subtask B: Multi-Class Authorship Detection
============================================
A machine learning tool to classify code snippets by author:
- Human
- 10 LLM families: DeepSeek-AI, Qwen, 01-ai, BigCode, Gemma, Phi, 
  Meta-LLaMA, IBM-Granite, Mistral, OpenAI

Usage:
    Training:   python train_subtask_b.py --train
    Testing:    python train_subtask_b.py --test
    Validation: python train_subtask_b.py --validate
    Prediction: python train_subtask_b.py --predict
    Auto:       python train_subtask_b.py (trains if no model exists, then predicts)
"""

import pandas as pd
import numpy as np
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
INPUT_FILE = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_B\task_b_training_set.parquet"
TEST_FILE = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_B\task_b_test_set_sample.parquet"
VALIDATION_FILE = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_B\task_b_validation_set.parquet"
MODEL_OUTPUT = r"C:\Users\gnand\OneDrive\Desktop\project\SUB_TASK_B\authorship_model.pkl"

# Training parameters
RANDOM_STATE = 42
N_ESTIMATORS = 250
MAX_DEPTH = 35

# Label definitions (11-class authorship classification)
LABEL_MAP = {
    0: 'human',
    1: 'DeepSeek-AI',
    2: 'Qwen',
    3: '01-ai',
    4: 'BigCode',
    5: 'Gemma',
    6: 'Phi',
    7: 'Meta-LLaMA',
    8: 'IBM-Granite',
    9: 'Mistral',
    10: 'OpenAI'
}

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(code: str) -> np.ndarray:
    """Extract 50 statistical and structural features from code snippet for authorship detection."""
    if not isinstance(code, str) or not code:
        return np.zeros(50)
    
    lines = code.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    
    features = []
    
    # BASIC METRICS (6 features)
    features.append(len(lines))
    features.append(len(code))
    features.append(len(code) / max(len(lines), 1))
    features.append(max([len(l) for l in lines], default=0))
    features.append(len(non_empty_lines) / max(len(lines), 1))
    features.append(np.std([len(l) for l in lines]) if len(lines) > 1 else 0)
    
    # WHITESPACE & FORMATTING (6 features)
    features.append(sum([1 for l in lines if l.startswith(' ')]) / max(len(lines), 1))
    features.append(sum([1 for l in lines if l.startswith('\t')]) / max(len(lines), 1))
    features.append(1 if '\t' in code else 0)
    features.append(1 if re.search(r'\s+$', code, re.MULTILINE) else 0)
    features.append(code.count(' ') / max(len(code), 1))
    features.append(code.count('\n\n') / max(len(lines), 1))  # Blank line density
    
    # COMMENTS (6 features)
    comment_count = len(re.findall(r'//|/\*|#|"""', code))
    features.append(comment_count)
    features.append(comment_count / max(len(lines), 1))
    features.append(1 if re.search(r'Example:|Usage:|Returns:|Args:|Parameters:', code) else 0)
    docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))
    features.append(docstring_count)
    features.append(len(re.findall(r'//.*TODO|#.*TODO|FIXME|HACK|XXX', code)))
    features.append(len(re.findall(r'//.*\w{20,}|#.*\w{20,}', code)))  # Long comments
    
    # NAMING CONVENTIONS (7 features)
    features.append(len(re.findall(r'[a-z]+[A-Z]', code)))  # camelCase
    features.append(len(re.findall(r'[a-z]+_[a-z]+', code)))  # snake_case
    features.append(len(re.findall(r'\b[a-z]\b', code)))  # Single char variables
    features.append(len(re.findall(r'[a-z]{10,}', code)))  # Long variable names
    features.append(len(re.findall(r'\b[A-Z][a-z]+[A-Z]', code)))  # PascalCase
    features.append(len(re.findall(r'\b[A-Z_]{2,}\b', code)))  # SCREAMING_SNAKE_CASE
    features.append(len(re.findall(r'\bvar\d+\b|\btemp\d+\b|\bx\d+\b', code)))  # Generic names
    
    # TYPE HINTS & DOCUMENTATION (4 features)
    features.append(1 if re.search(r':\s*(int|str|float|bool|List|Dict|Optional|Any|void)', code) else 0)
    features.append(len(re.findall(r'->', code)))  # Return type hints
    features.append(len(re.findall(r'@\w+', code)))  # Decorators/annotations
    features.append(len(re.findall(r'<\w+>', code)))  # Generic type parameters
    
    # CODE STRUCTURE (6 features)
    features.append(len(re.findall(r'def |function |class |interface |struct ', code)))
    features.append(1 if re.search(r'try|except|catch|finally|throw', code) else 0)
    features.append(1 if re.search(r'print|console\.log|logger|cout|System\.out', code) else 0)
    features.append(len(re.findall(r'import |require\(|from .* import|#include|using ', code)))
    features.append(len(re.findall(r'return ', code)))
    features.append(len(re.findall(r'assert |assertEqual|expect\(', code)))  # Testing code
    
    # CONTROL FLOW (5 features)
    features.append(count_nesting(code))
    features.append(len(re.findall(r'\bif\b|\bwhile\b|\bfor\b', code)))
    features.append(len(re.findall(r'\belse\b|\belif\b', code)))
    features.append(len(re.findall(r'break|continue', code)))
    features.append(len(re.findall(r'switch|case|match', code)))  # Switch statements
    
    # INDENTATION CONSISTENCY (3 features)
    indents = [len(l) - len(l.lstrip()) for l in non_empty_lines if l.strip()]
    if len(indents) > 1:
        indent_diffs = [abs(indents[i] - indents[i-1]) for i in range(1, len(indents))]
        unique_diffs = len(set([d for d in indent_diffs if d > 0]))
        features.append(1 if unique_diffs <= 2 else 0)
        features.append(np.std(indents) if len(indents) > 1 else 0)
        features.append(np.mean(indents) if len(indents) > 0 else 0)
    else:
        features.extend([0, 0, 0])
    
    # OPERATORS & SYMBOLS (5 features)
    features.append(len(re.findall(r'===|!==|==|!=|<=|>=', code)))
    features.append(len(re.findall(r'\{', code)))  # Opening braces
    features.append(len(re.findall(r'\(', code)))  # Opening parens
    features.append(code.count(';') / max(len(lines), 1))
    features.append(len(re.findall(r'\[', code)))  # Opening brackets
    
    # CODE QUALITY INDICATORS (6 features)
    features.append(len(re.findall(r'[A-Z]{2,}', code)))  # CONSTANTS
    features.append(1 if re.search(r'lambda |=>|\bmap\b|\bfilter\b|\breduce\b', code) else 0)
    features.append(1 if len(code) > 500 and comment_count == 0 else 0)
    features.append(len(re.findall(r'async|await|Promise', code)))
    features.append(len(re.findall(r'yield|generator', code)))
    features.append(1 if re.search(r'TODO|FIXME|HACK|XXX|BUG', code) else 0)
    
    # LLM-SPECIFIC PATTERNS (6 features)
    features.append(1 if re.search(r'# Example usage|# Usage example', code, re.IGNORECASE) else 0)
    features.append(1 if re.search(r'Here\'s|Here is', code, re.IGNORECASE) else 0)
    features.append(len(re.findall(r'Note:|Important:|Warning:', code)))
    features.append(1 if re.search(r'This \w+ (will|can|should|must)', code) else 0)
    features.append(len(re.findall(r'```', code)))  # Markdown code blocks
    features.append(1 if re.search(r'Output:|Result:|Explanation:', code) else 0)
    
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
    print(f"F1 Score (Macro):        {f1_macro*100:6.2f}%  <- MAIN METRIC")
    print(f"F1 Score (Weighted):     {f1_weighted*100:6.2f}%")
    print(f"F1 Score (Micro):        {f1_micro*100:6.2f}%")
    print(f"{'='*58}")
    
    # F1 scores per class
    print(f"\nF1 Score by Class:")
    for i, f1 in enumerate(f1_per_class):
        bar = "█" * int(f1 * 50)
        print(f"  {LABEL_MAP[i]:15} {f1*100:6.2f}% {bar}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        y, y_pred, 
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
        digits=4
    ))
    
    # Confusion matrix (simplified for 11 classes)
    print(f"\nConfusion Matrix (showing top errors):")
    cm = confusion_matrix(y, y_pred)
    
    # Find top misclassifications
    misclassifications = []
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            if i != j and cm[i][j] > 0:
                misclassifications.append((cm[i][j], LABEL_MAP[i], LABEL_MAP[j]))
    
    misclassifications.sort(reverse=True)
    print(f"\nTop 10 Misclassifications:")
    for count, actual, predicted in misclassifications[:10]:
        print(f"  {count:4} samples: {actual:15} → {predicted:15}")
    
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
    print("TRAINING MODE - AUTHORSHIP DETECTION (SUBTASK B)")
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
            print(f"   {label:15} {count:6,} ({percentage:5.2f}%) {bar}")
        
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
            verbose=1,
            class_weight='balanced'  # Handle imbalanced classes
        )
        model.fit(X_train, y_train)
        print(f"   ✓ Model trained!")
        
        # Calculate training metrics
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
        print(f"   F1 Score (Macro):        {f1_macro*100:6.2f}%  <- MAIN METRIC")
        print(f"   F1 Score (Weighted):     {f1_weighted*100:6.2f}%")
        print(f"   F1 Score (Micro):        {f1_micro*100:6.2f}%")
        print(f"   {'='*58}")
        
        print(f"\n   F1 Score by Class:")
        for i, f1 in enumerate(f1_per_class):
            bar = "█" * int(f1 * 50)
            print(f"   {LABEL_MAP[i]:15} {f1*100:6.2f}% {bar}")
        
        # Save model
        print(f"\n   Saving model to: {MODEL_OUTPUT}")
        os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
        
        feature_names = [
            "Total lines", "Total chars", "Avg chars/line", "Max line length", "Non-empty ratio", "Line length std",
            "Indented lines ratio", "Tab indented ratio", "Uses tabs", "Trailing whitespace", "Space density", "Blank line density",
            "Comment count", "Comment density", "Has structured docs", "Docstring count", "TODO markers", "Long comments",
            "camelCase count", "snake_case count", "Single letter vars", "Long var names", "PascalCase count", "SCREAMING_CASE", "Generic names",
            "Has type hints", "Return type hints", "Decorators/annotations", "Generic types",
            "Function/class defs", "Has error handling", "Has logging", "Import statements", "Return statements", "Has tests",
            "Max nesting", "If/while/for count", "Else/elif count", "Break/continue", "Switch statements",
            "Consistent indent", "Indent variance", "Avg indent",
            "Equality checks", "Brace count", "Paren count", "Semicolons/line", "Bracket count",
            "Constants/acronyms", "Functional programming", "Large uncommented", "Async patterns", "Generators", "Has markers",
            "Example usage", "Here's pattern", "Note/Warning", "Instructional", "Markdown blocks", "Output labels"
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
            'subtask': 'B'
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
    print("TESTING MODE - AUTHORSHIP DETECTION (SUBTASK B)")
    print("=" * 70)
    
    if not os.path.exists(MODEL_OUTPUT):
        print(f"\n✗ Error: Model file not found: {MODEL_OUTPUT}")
        print("Please train the model first: python train_subtask_b.py --train")
        return False
    
    if not os.path.exists(TEST_FILE):
        print(f"\n✗ Error: Test file not found: {TEST_FILE}")
        return False
    
    try:
        # Load model
        print(f"\n[1/3] Loading trained model...")
        model_data = joblib.load(MODEL_OUTPUT)
        model = model_data['model']
        print(f"   ✓ Model loaded!")
        print(f"   Training date: {model_data.get('training_date', 'Unknown')}")
        print(f"   Training accuracy: {model_data.get('accuracy', 0)*100:.2f}%")
        print(f"   Training Macro F1: {model_data.get('f1_macro', 0)*100:.2f}%")
        
        # Load test data
        print(f"\n[2/3] Loading test data...")
        print(f"   Path: {TEST_FILE}")
        df_test = pd.read_parquet(TEST_FILE)
        print(f"   ✓ Loaded {len(df_test):,} test samples")
        
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
    print("VALIDATION MODE - AUTHORSHIP DETECTION (SUBTASK B)")
    print("=" * 70)
    
    if not os.path.exists(MODEL_OUTPUT):
        print(f"\n✗ Error: Model file not found: {MODEL_OUTPUT}")
        print("Please train the model first: python train_subtask_b.py --train")
        return False
    
    if not os.path.exists(VALIDATION_FILE):
        print(f"\n✗ Error: Validation file not found: {VALIDATION_FILE}")
        return False
    
    try:
        # Load model
        print(f"\n[1/3] Loading trained model...")
        model_data = joblib.load(MODEL_OUTPUT)
        model = model_data['model']
        print(f"   ✓ Model loaded!")
        print(f"   Training date: {model_data.get('training_date', 'Unknown')}")
        print(f"   Training accuracy: {model_data.get('accuracy', 0)*100:.2f}%")
        print(f"   Training Macro F1: {model_data.get('f1_macro', 0)*100:.2f}%")
        
        # Load validation data
        print(f"\n[2/3] Loading validation data...")
        print(f"   Path: {VALIDATION_FILE}")
        df_val = pd.read_parquet(VALIDATION_FILE)
        print(f"   ✓ Loaded {len(df_val):,} validation samples")
        
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
        print("Please run training mode first: python train_subtask_b.py --train")
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
        print("AUTHORSHIP CLASSIFICATION RESULT")
        print("=" * 60)
        print(f"\nPredicted Author: {class_name.upper()}")
        print(f"Confidence: {probabilities[prediction]*100:.2f}%")
        
        print(f"\nTop 5 Predictions:")
        top_indices = np.argsort(probabilities)[::-1][:5]
        for idx in top_indices:
            label_name = label_map[idx]
            prob = probabilities[idx]
            bar = "█" * int(prob * 50)
            print(f"  {label_name:15} {prob*100:6.2f}% {bar}")
        
        print(f"\nModel Information:")
        if 'training_date' in model_data:
            print(f"  Trained: {model_data['training_date']}")
        if 'total_samples' in model_data:
            print(f"  Training samples: {model_data['total_samples']:,}")
        if 'f1_macro' in model_data:
            print(f"  Macro F1: {model_data['f1_macro']*100:.2f}%")
        
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
    print("PREDICTION MODE - AUTHORSHIP DETECTION (SUBTASK B)")
    print("=" * 70)
    
    test_code = """
def fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
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
    print("AUTHORSHIP DETECTOR - SUBTASK B")
    print("=" * 70)
    print(f"Version: 1.0")
    print(f"Model: Random Forest Classifier")
    print(f"Classes: {', '.join(LABEL_MAP.values())}")
    print(f"Target Metric: Macro F1-Score")
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
            print("  Training:    python train_subtask_b.py --train")
            print("  Testing:     python train_subtask_b.py --test")
            print("  Validation:  python train_subtask_b.py --validate")
            print("  Prediction:  python train_subtask_b.py --predict")
            print("  Auto mode:   python train_subtask_b.py")
            print("  Help:        python train_subtask_b.py --help")
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