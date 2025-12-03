# Simple Immediate Processing - No Storage Required

## Option: Process File Immediately

Instead of storing files, analyze them during the upload request. This works perfectly on Google Cloud without needing GCS.

### How It Works

1. User uploads CSV file
2. Flask receives file in memory
3. Save temporarily to `/tmp` for processing
4. Analyze immediately
5. Return results to user
6. File discarded (no need to persist)

### Implementation

Add this new endpoint to `flask_server.py`:

```python
@app.route('/api/upload-and-analyze', methods=['POST'])
def upload_and_analyze():
    """Upload CSV and analyze immediately - no storage needed"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
        
        # Save to /tmp temporarily (works on GCloud)
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)
        
        # Validate structure
        is_valid, message = validate_csv_structure(temp_path)
        if not is_valid:
            os.remove(temp_path)
            return jsonify({'error': message}), 400
        
        # Process immediately
        if not PROCESSING_AVAILABLE:
            os.remove(temp_path)
            return jsonify({'error': 'Processing module not available'}), 500
        
        from processing import MovementIdentifier
        
        identifier = MovementIdentifier()
        identifier.load_data(temp_path)
        
        # Analyze both sides
        left_segments = identifier.detect_movement_segments('left', min_duration=5)
        right_segments = identifier.detect_movement_segments('right', min_duration=5)
        body_segments = identifier.detect_body_movement_segments(min_duration=5)
        
        left_segments = sorted(left_segments + body_segments, key=lambda x: x['start_frame'])
        right_segments = sorted(right_segments + body_segments, key=lambda x: x['start_frame'])
        
        # Get coordinate data for visualization
        left_wrist = identifier.get_joint_coordinates('wrist', 'left')
        right_wrist = identifier.get_joint_coordinates('wrist', 'right')
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Return analysis results immediately
        return jsonify({
            'success': True,
            'filename': file.filename,
            'left_segments': left_segments[:20],  # Limit for response size
            'right_segments': right_segments[:20],
            'total_left_segments': len(left_segments),
            'total_right_segments': len(right_segments),
            'total_frames': len(identifier.data) if identifier.data is not None else 0
        })
        
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
```

### Benefits

✓ **No GCS setup required** - Works immediately  
✓ **No storage costs** - Files not persisted  
✓ **No file management** - No cleanup needed  
✓ **Works on free tier** - No additional services  
✓ **Secure** - Files not stored permanently  

### Limitations

✗ Can't re-analyze same file later (must re-upload)  
✗ Longer response time (processing happens during upload)  
✗ Request timeout if file is very large (10 minutes max on App Engine)

### Frontend Change Needed

Update your upload form to call the new endpoint:

```javascript
// Instead of two-step upload then analyze:
fetch('/api/upload', ...).then(() => fetch('/analyze-uploaded/...'))

// Use single-step:
fetch('/api/upload-and-analyze', {
    method: 'POST',
    body: formData
}).then(response => response.json())
  .then(data => {
      // Display results immediately
      console.log(data.left_segments);
      console.log(data.right_segments);
  });
```

## Recommendation

**For your use case**: Use immediate processing (no storage). Your CSV files are motion capture data that users want analyzed once. There's no need to store them.

If you later need file history or re-analysis, then set up GCS.
