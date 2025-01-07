#!/bin/bash

# Function to count lines for a specific file type
count_lines() {
    echo "Counting lines for $1 files:"
    find . -name "*.$1" -type f ! -path "*/node_modules/*" -exec wc -l {} + | awk '{total += $1} END {print total " lines of " extension}' extension="$1"
    echo ""
}

# Main execution
echo "Line count by file type:"
echo "------------------------"
count_lines js
count_lines css
count_lines ts
count_lines tsx
count_lines py
count_lines yml
count_lines yaml

