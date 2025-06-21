#!/bin/bash
# Cleanup script for BERT demo build artifacts

echo "🧹 BERT Demo Cleanup Script"
echo "=========================="

# Function to show size of directory before deletion
show_size() {
    if [ -d "$1" ]; then
        SIZE=$(du -sh "$1" 2>/dev/null | cut -f1)
        echo "  📁 $1 ($SIZE)"
    fi
}

# Check what would be deleted
echo "📊 Directories to clean:"
show_size "quicktest_output"
show_size "quicktest_output_folded"
show_size "finn_output"
show_size "__pycache__"
show_size ".pytest_cache"

# Check for generated files
echo ""
echo "📄 Files to clean:"
for file in *.json *.npy *.npz *.onnx *.log; do
    if [ -f "$file" ]; then
        SIZE=$(du -sh "$file" 2>/dev/null | cut -f1)
        echo "  📄 $file ($SIZE)"
    fi
done

# Ask for confirmation
echo ""
read -p "🤔 Delete all these files and directories? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Cleaning up..."
    
    # Remove directories
    rm -rf quicktest_output quicktest_output_folded finn_output
    rm -rf __pycache__ .pytest_cache
    
    # Remove generated files (but keep source files)
    rm -f safe_folding*.json compatible_folding*.json
    rm -f *.npy *.npz
    
    # Optional: Remove any generated ONNX models in current directory
    # (but not in subdirectories where they might be needed)
    rm -f bert_model.onnx
    
    # Remove log files
    rm -f *.log
    
    echo "✅ Cleanup complete!"
    
    # Show remaining files
    echo ""
    echo "📋 Remaining files in directory:"
    ls -la | grep -v "^d" | tail -n +2
else
    echo "❌ Cleanup cancelled"
fi