#!/bin/bash

echo "=========================================="
echo "Deepfake Detection App - Setup Verification"
echo "=========================================="
echo ""

# Check Python version
echo "1. Checking Python version..."
python --version
echo ""

# Check if all source files compile
echo "2. Checking Python syntax..."
python -m py_compile app.py src/model.py src/inference.py src/train.py && echo "   ✓ All files compile successfully" || echo "   ✗ Syntax errors found"
echo ""

# List dependencies
echo "3. Checking dependencies..."
echo "   Required packages in requirements.txt:"
cat requirements.txt | grep -v "^#" | grep -v "^$"
echo ""

# Check file structure
echo "4. Checking file structure..."
echo "   Core files:"
[ -f app.py ] && echo "   ✓ app.py" || echo "   ✗ app.py missing"
[ -f requirements.txt ] && echo "   ✓ requirements.txt" || echo "   ✗ requirements.txt missing"
[ -f .gitignore ] && echo "   ✓ .gitignore" || echo "   ✗ .gitignore missing"
echo ""
echo "   Documentation:"
[ -f readme.md ] && echo "   ✓ readme.md" || echo "   ✗ readme.md missing"
[ -f QUICKSTART.md ] && echo "   ✓ QUICKSTART.md" || echo "   ✗ QUICKSTART.md missing"
[ -f CHANGES.md ] && echo "   ✓ CHANGES.md" || echo "   ✗ CHANGES.md missing"
echo ""
echo "   Source files:"
[ -f src/model.py ] && echo "   ✓ src/model.py" || echo "   ✗ src/model.py missing"
[ -f src/inference.py ] && echo "   ✓ src/inference.py" || echo "   ✗ src/inference.py missing"
[ -f src/train.py ] && echo "   ✓ src/train.py" || echo "   ✗ src/train.py missing"
echo ""
echo "   Tests:"
[ -f test_model.py ] && echo "   ✓ test_model.py" || echo "   ✗ test_model.py missing"
echo ""

# Test model architecture
echo "5. Testing model architecture..."
python test_model.py 2>&1 | grep -E "(All tests passed|TEST SUMMARY)" -A 5
echo ""

echo "=========================================="
echo "Verification Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Prepare dataset in data/ directory"
echo "3. Train model: python src/train.py"
echo "4. Run app: streamlit run app.py"
echo ""
echo "For detailed instructions, see QUICKSTART.md"
