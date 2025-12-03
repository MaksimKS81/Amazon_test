"""
Local development server runner
Ensures proper environment setup before starting Flask
"""
import os
import sys

# Ensure uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')
    print("✓ Created uploads directory")

# Ensure templates directory exists
if not os.path.exists('templates'):
    os.makedirs('templates')
    print("✓ Created templates directory")

# Check required files
required_files = ['flask_server.py', 'processing.py', 'requirements.txt', 'app.yaml']
for file in required_files:
    if not os.path.exists(file):
        print(f"✗ Warning: {file} not found")

print("\nStarting local development server...")
print("Access at: http://127.0.0.1:5000")
print("Press Ctrl+C to stop\n")

# Import and run flask app
from flask_server import app
app.run(debug=True, host='127.0.0.1', port=5000)
