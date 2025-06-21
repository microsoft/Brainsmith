#!/bin/bash
# Deep cleanup script for BERT demo - removes ALL generated artifacts

echo "🧨 BERT Demo DEEP CLEAN Script"
echo "=============================="
echo "⚠️  WARNING: This will remove ALL generated files and build artifacts!"
echo ""

# Function to show size of directory before deletion
show_size() {
    if [ -d "$1" ]; then
        SIZE=$(du -sh "$1" 2>/dev/null | cut -f1)
        echo "  📁 $1 ($SIZE)"
    else
        echo "  📁 $1 (not found)"
    fi
}

# Show what will be deleted
echo "📊 Everything that will be deleted:"
echo ""
echo "Build outputs:"
show_size "quicktest_output*"
show_size "finn_output"
show_size "output"
show_size "outputs"

echo ""
echo "Generated configs:"
ls -la *.json 2>/dev/null | awk '{print "  📄 " $9 " (" $5 " bytes)"}'

echo ""
echo "Model files:"
ls -la *.onnx 2>/dev/null | awk '{print "  📄 " $9 " (" $5 " bytes)"}'

echo ""
echo "Data files:"
ls -la *.npy *.npz 2>/dev/null | awk '{print "  📄 " $9 " (" $5 " bytes)"}'

echo ""
echo "Cache and temp:"
show_size "__pycache__"
show_size ".pytest_cache"
show_size "*.pyc"

echo ""
echo "Logs:"
ls -la *.log 2>/dev/null | awk '{print "  📄 " $9 " (" $5 " bytes)"}'

# Also check container build directory
if [ -n "$BSMITH_BUILD_DIR" ]; then
    echo ""
    echo "Container build artifacts:"
    show_size "$BSMITH_BUILD_DIR/bert_demo*"
fi

# Double confirmation for deep clean
echo ""
echo "⚠️  This is a DEEP CLEAN - all generated files will be removed!"
read -p "🤔 Are you sure you want to delete ALL of this? (type 'yes' to confirm) " -r
echo ""

if [[ $REPLY == "yes" ]]; then
    echo "🗑️  Deep cleaning..."
    
    # Remove all build directories
    rm -rf quicktest_output* finn_output output outputs
    
    # Remove all generated JSON configs
    rm -f *.json
    
    # Remove all ONNX models
    rm -f *.onnx
    
    # Remove all numpy files
    rm -f *.npy *.npz
    
    # Remove cache directories
    rm -rf __pycache__ .pytest_cache
    find . -name "*.pyc" -delete
    
    # Remove all log files
    rm -f *.log
    
    # Clean container build directory if accessible
    if [ -n "$BSMITH_BUILD_DIR" ] && [ -d "$BSMITH_BUILD_DIR" ]; then
        echo "🐋 Cleaning container build artifacts..."
        rm -rf $BSMITH_BUILD_DIR/bert_demo*
    fi
    
    echo "✅ Deep clean complete!"
    
    # Show what's left
    echo ""
    echo "📋 Remaining files (only source code should remain):"
    ls -la
else
    echo "❌ Deep clean cancelled"
fi