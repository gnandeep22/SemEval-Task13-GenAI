"""
Flask Backend Server for AI Code Detection System
==================================================
Serves predictions for all three subtasks through REST API

Usage:
    python app.py

Endpoints:
    POST /predict_a - Subtask A (Binary Detection)
    POST /predict_b - Subtask B (Authorship)
    POST /predict_c - Subtask C (Hybrid Detection)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import re
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# ============================================================================
# MODEL PATHS - Models are in parent directory's subtask folders
# ============================================================================

# Get the parent directory (project folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # Go up one level from website folder

MODEL_A_PATH = os.path.join(PROJECT_DIR, 'SUB_TASK_A', 'binary_code_model.pkl')
MODEL_B_PATH = os.path.join(PROJECT_DIR, 'SUB_TASK_B', 'authorship_model.pkl')
MODEL_C_PATH = os.path.join(PROJECT_DIR, 'SUB_TASK_C', 'hybrid_code_model.pkl')

print(f"Looking for models in:")
print(f"  Model A: {MODEL_A_PATH}")
print(f"  Model B: {MODEL_B_PATH}")
print(f"  Model C: {MODEL_C_PATH}")

# Load models at startup
models = {}

def load_models():
    """Load all available models"""
    if os.path.exists(MODEL_A_PATH):
        models['A'] = joblib.load(MODEL_A_PATH)
        print("✓ Model A loaded successfully")
    else:
        print(f"⚠ Model A not found at {MODEL_A_PATH}")
    
    if os.path.exists(MODEL_B_PATH):
        models['B'] = joblib.load(MODEL_B_PATH)
        print("✓ Model B loaded successfully")
    else:
        print(f"⚠ Model B not found at {MODEL_B_PATH}")
    
    if os.path.exists(MODEL_C_PATH):
        models['C'] = joblib.load(MODEL_C_PATH)
        print("✓ Model C loaded successfully")
    else:
        print(f"⚠ Model C not found at {MODEL_C_PATH}")

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def count_nesting(code: str) -> int:
    """Calculate maximum nesting level"""
    max_depth = 0
    current_depth = 0
    for char in code:
        if char in '{([':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char in '})]':
            current_depth = max(0, current_depth - 1)
    return max_depth


def extract_features_a(code: str) -> np.ndarray:
    """Extract 40 features for Task A"""
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
        features.extend([0, 0])
    
    # OPERATORS & SYMBOLS (4)
    features.append(len(re.findall(r'===|!==|==|!=|<=|>=', code)))
    features.append(len(re.findall(r'\{', code)))
    features.append(len(re.findall(r'\(', code)))
    features.append(code.count(';') / max(len(lines), 1))
    
    # CODE QUALITY INDICATORS (3)
    features.append(len(re.findall(r'[A-Z]{2,}', code)))
    features.append(1 if re.search(r'lambda |=>|\bmap\b|\bfilter\b|\breduce\b', code) else 0)
    features.append(1 if len(code) > 500 and comment_count == 0 else 0)
    
    return np.array(features)


def extract_features_b(code: str) -> np.ndarray:
    """Extract 50 features for Task B"""
    if not isinstance(code, str) or not code:
        return np.zeros(50)
    
    lines = code.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    features = []
    
    # BASIC METRICS (6)
    features.append(len(lines))
    features.append(len(code))
    features.append(len(code) / max(len(lines), 1))
    features.append(max([len(l) for l in lines], default=0))
    features.append(len(non_empty_lines) / max(len(lines), 1))
    features.append(np.std([len(l) for l in lines]) if len(lines) > 1 else 0)
    
    # WHITESPACE & FORMATTING (6)
    features.append(sum([1 for l in lines if l.startswith(' ')]) / max(len(lines), 1))
    features.append(sum([1 for l in lines if l.startswith('\t')]) / max(len(lines), 1))
    features.append(1 if '\t' in code else 0)
    features.append(1 if re.search(r'\s+$', code, re.MULTILINE) else 0)
    features.append(code.count(' ') / max(len(code), 1))
    features.append(code.count('\n\n') / max(len(lines), 1))
    
    # COMMENTS (6)
    comment_count = len(re.findall(r'//|/\*|#|"""', code))
    features.append(comment_count)
    features.append(comment_count / max(len(lines), 1))
    features.append(1 if re.search(r'Example:|Usage:|Returns:|Args:|Parameters:', code) else 0)
    docstring_count = len(re.findall(r'""".*?"""', code, re.DOTALL))
    features.append(docstring_count)
    features.append(len(re.findall(r'//.*TODO|#.*TODO|FIXME|HACK|XXX', code)))
    features.append(len(re.findall(r'//.*\w{20,}|#.*\w{20,}', code)))
    
    # NAMING CONVENTIONS (7)
    features.append(len(re.findall(r'[a-z]+[A-Z]', code)))
    features.append(len(re.findall(r'[a-z]+_[a-z]+', code)))
    features.append(len(re.findall(r'\b[a-z]\b', code)))
    features.append(len(re.findall(r'[a-z]{10,}', code)))
    features.append(len(re.findall(r'\b[A-Z][a-z]+[A-Z]', code)))
    features.append(len(re.findall(r'\b[A-Z_]{2,}\b', code)))
    features.append(len(re.findall(r'\bvar\d+\b|\btemp\d+\b|\bx\d+\b', code)))
    
    # TYPE HINTS & DOCUMENTATION (4)
    features.append(1 if re.search(r':\s*(int|str|float|bool|List|Dict|Optional|Any|void)', code) else 0)
    features.append(len(re.findall(r'->', code)))
    features.append(len(re.findall(r'@\w+', code)))
    features.append(len(re.findall(r'<\w+>', code)))
    
    # CODE STRUCTURE (6)
    features.append(len(re.findall(r'def |function |class |interface |struct ', code)))
    features.append(1 if re.search(r'try|except|catch|finally|throw', code) else 0)
    features.append(1 if re.search(r'print|console\.log|logger|cout|System\.out', code) else 0)
    features.append(len(re.findall(r'import |require\(|from .* import|#include|using ', code)))
    features.append(len(re.findall(r'return ', code)))
    features.append(len(re.findall(r'assert |assertEqual|expect\(', code)))
    
    # CONTROL FLOW (5)
    features.append(count_nesting(code))
    features.append(len(re.findall(r'\bif\b|\bwhile\b|\bfor\b', code)))
    features.append(len(re.findall(r'\belse\b|\belif\b', code)))
    features.append(len(re.findall(r'break|continue', code)))
    features.append(len(re.findall(r'switch|case|match', code)))
    
    # INDENTATION CONSISTENCY (3)
    indents = [len(l) - len(l.lstrip()) for l in non_empty_lines if l.strip()]
    if len(indents) > 1:
        indent_diffs = [abs(indents[i] - indents[i-1]) for i in range(1, len(indents))]
        unique_diffs = len(set([d for d in indent_diffs if d > 0]))
        features.append(1 if unique_diffs <= 2 else 0)
        features.append(np.std(indents) if len(indents) > 1 else 0)
        features.append(np.mean(indents) if len(indents) > 0 else 0)
    else:
        features.extend([0, 0, 0])
    
    # OPERATORS & SYMBOLS (5)
    features.append(len(re.findall(r'===|!==|==|!=|<=|>=', code)))
    features.append(len(re.findall(r'\{', code)))
    features.append(len(re.findall(r'\(', code)))
    features.append(code.count(';') / max(len(lines), 1))
    features.append(len(re.findall(r'\[', code)))
    
    # CODE QUALITY INDICATORS (6)
    features.append(len(re.findall(r'[A-Z]{2,}', code)))
    features.append(1 if re.search(r'lambda |=>|\bmap\b|\bfilter\b|\breduce\b', code) else 0)
    features.append(1 if len(code) > 500 and comment_count == 0 else 0)
    features.append(len(re.findall(r'async|await|Promise', code)))
    features.append(len(re.findall(r'yield|generator', code)))
    features.append(1 if re.search(r'TODO|FIXME|HACK|XXX|BUG', code) else 0)
    
    # LLM-SPECIFIC PATTERNS (6)
    features.append(1 if re.search(r'# Example usage|# Usage example', code, re.IGNORECASE) else 0)
    features.append(1 if re.search(r'Here\'s|Here is', code, re.IGNORECASE) else 0)
    features.append(len(re.findall(r'Note:|Important:|Warning:', code)))
    features.append(1 if re.search(r'This \w+ (will|can|should|must)', code) else 0)
    features.append(len(re.findall(r'```', code)))
    features.append(1 if re.search(r'Output:|Result:|Explanation:', code) else 0)
    
    return np.array(features)


def extract_features_c(code: str) -> np.ndarray:
    """Extract 35 features for Task C"""
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
    
    # WHITESPACE PATTERNS (3)
    features.append(sum([1 for l in lines if l.startswith(' ')]) / max(len(lines), 1))
    features.append(1 if '\t' in code else 0)
    features.append(1 if re.search(r'\s+$', code, re.MULTILINE) else 0)
    
    # COMMENTS (3)
    comment_count = len(re.findall(r'//|/\*|#|"""', code))
    features.append(comment_count)
    features.append(comment_count / max(len(lines), 1))
    features.append(1 if re.search(r'Example:|Usage:|Returns:|Args:', code) else 0)
    
    # NAMING CONVENTIONS (4)
    features.append(len(re.findall(r'[a-z]+[A-Z]', code)))
    features.append(len(re.findall(r'[a-z]+_[a-z]+', code)))
    features.append(len(re.findall(r'\b[a-z]\b', code)))
    features.append(len(re.findall(r'[a-z]{8,}', code)))
    
    # TYPE HINTS & DOCUMENTATION (2)
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
    
    # INDENTATION CONSISTENCY (1)
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
    
    # CODE MARKERS (1)
    features.append(1 if re.search(r'TODO|FIXME|HACK|XXX', code) else 0)
    
    # ADVANCED PATTERNS (5)
    features.append(len(re.findall(r'import |require\(|from .* import', code)))
    features.append(len(re.findall(r'===|!==|==|!=', code)))
    features.append(len(re.findall(r'\{|\}', code)))
    features.append(len(re.findall(r'\(|\)', code)))
    features.append(1 if re.search(r'async|await|Promise', code) else 0)
    
    # CODE QUALITY INDICATORS (5)
    features.append(len(re.findall(r'[A-Z]{2,}', code)))
    features.append(len(non_empty_lines) / max(len(lines), 1))
    features.append(1 if re.search(r'lambda |=>|\bmap\b|\bfilter\b', code) else 0)
    features.append(code.count(';') / max(len(lines), 1))
    features.append(1 if len(code) > 500 and comment_count == 0 else 0)
    
    return np.array(features)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/predict_a', methods=['POST'])
def predict_a():
    """Subtask A: Binary Detection"""
    try:
        if 'A' not in models:
            return jsonify({'error': 'Model A not loaded'}), 500
        
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Extract features
        features = extract_features_a(code).reshape(1, -1)
        
        # Make prediction
        model_data = models['A']
        model = model_data['model']
        label_map = model_data['label_map']
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Format response
        result = {
            'prediction': label_map[prediction],
            'confidence': round(probabilities[prediction] * 100, 2),
            'probabilities': {
                label_map[i]: round(prob * 100, 2) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_b', methods=['POST'])
def predict_b():
    """Subtask B: Authorship Detection"""
    try:
        if 'B' not in models:
            return jsonify({'error': 'Model B not loaded'}), 500
        
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Extract features
        features = extract_features_b(code).reshape(1, -1)
        
        # Make prediction
        model_data = models['B']
        model = model_data['model']
        label_map = model_data['label_map']
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Format response
        result = {
            'prediction': label_map[prediction],
            'confidence': round(probabilities[prediction] * 100, 2),
            'probabilities': {
                label_map[i]: round(prob * 100, 2) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_c', methods=['POST'])
def predict_c():
    """Subtask C: Hybrid Detection"""
    try:
        if 'C' not in models:
            return jsonify({'error': 'Model C not loaded'}), 500
        
        data = request.json
        code = data.get('code', '')
        
        if not code:
            return jsonify({'error': 'No code provided'}), 400
        
        # Extract features
        features = extract_features_c(code).reshape(1, -1)
        
        # Make prediction
        model_data = models['C']
        model = model_data['model']
        label_map = model_data['label_map']
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Format response
        result = {
            'prediction': label_map[prediction],
            'confidence': round(probabilities[prediction] * 100, 2),
            'probabilities': {
                label_map[i]: round(prob * 100, 2) 
                for i, prob in enumerate(probabilities)
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'available_models': list(models.keys())
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("AI CODE DETECTION SYSTEM - FLASK SERVER")
    print("="*70)
    print("\nLoading models...")
    load_models()
    print(f"\nAvailable models: {list(models.keys())}")
    print("\nStarting Flask server on http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)