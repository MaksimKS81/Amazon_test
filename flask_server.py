import sys
import os
import logging

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required packages and provide helpful error messages
try:
    from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
    from werkzeug.utils import secure_filename
    logger.info("✓ Flask imported successfully")
except ImportError as e:
    logger.error("✗ Flask not found. Install with: pip install flask")
    sys.exit(1)

try:
    import pandas as pd
    logger.info("✓ Pandas imported successfully")
except ImportError as e:
    logger.error("✗ Pandas not found. Install with: pip install pandas")
    sys.exit(2)

try:
    import numpy as np
    logger.info("✓ NumPy imported successfully")
except ImportError as e:
    logger.error("✗ NumPy not found. Install with: pip install numpy")
    sys.exit(3)

try:
    import plotly.graph_objs as go
    import plotly.utils
    PLOTLY_AVAILABLE = True
    logger.info("✓ Plotly imported successfully")
except ImportError as e:
    logger.warning("⚠ Warning: Plotly not found. Some features will be limited.")
    logger.warning("  Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

import json

# Check for Google Cloud Storage (optional for local development)
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
    logger.info("✓ Google Cloud Storage available")
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("⚠ Google Cloud Storage not available - using local storage")

# Check for processing module availability
try:
    import processing
    PROCESSING_AVAILABLE = True
    logger.info("✓ Processing module available")
except ImportError:
    PROCESSING_AVAILABLE = False
    logger.warning("⚠ Warning: Processing module not available")

# Initialize Flask app with explicit template folder
app = Flask(__name__, template_folder='templates')

# Configuration - Use /tmp for all uploaded files (session-based storage)
IS_PRODUCTION = os.getenv('GAE_ENV', '').startswith('standard')
UPLOAD_FOLDER = '/tmp'  # Works on both local and Google Cloud
ALLOWED_EXTENSIONS = {'csv', 'txt'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Custom template filter for date formatting
@app.template_filter('timestamp_to_date')
def timestamp_to_date(timestamp):
    import datetime
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
app.secret_key = 'movement_analysis_upload_key_2024'  # For flash messages

# Ensure directories exist
for folder in ['templates', '/tmp']:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        logger.info(f"Created {folder} directory")

# File upload helper functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_path(filename):
    """Get full file path in /tmp directory"""
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

def save_uploaded_file(file, filename):
    """Save uploaded file to /tmp directory"""
    try:
        filepath = get_file_path(filename)
        file.save(filepath)
        logger.info(f"✓ Saved {filename} to {filepath}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to save {filename}: {e}")
        return False

def list_uploaded_files():
    """List all uploaded files in /tmp directory"""
    uploads_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(uploads_dir):
        return [f for f in os.listdir(uploads_dir) if allowed_file(f)]
    return []

def validate_csv_structure(filepath):
    """Validate that uploaded CSV has expected structure for movement data"""
    try:
        df = pd.read_csv(filepath)
        # Check if it has reasonable number of columns (motion capture data typically has many columns)
        if len(df.columns) < 10:
            return False, "CSV file should have motion capture data with multiple coordinate columns"
        
        # Check if it has data rows
        if len(df) < 10:
            return False, "CSV file should have at least 10 frames of movement data"
            
        return True, "Valid movement data file"
    except Exception as e:
        return False, f"Error reading CSV file: {str(e)}"

# Sample data generation function
def generate_sample_data():
    """Generate sample data for visualization"""
    try:
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = np.cumsum(np.random.randn(100)) + 100
        return dates, values
    except Exception as e:
        print(f"Error generating sample data: {e}")
        return [], []

def create_fallback_data():
    """Create simple data when Plotly is not available"""
    import datetime
    dates = []
    values = []
    base_date = datetime.datetime(2023, 1, 1)
    
    for i in range(10):
        dates.append((base_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))
        values.append(100 + i * 5 + np.random.randn() * 2)
    
    return {
        'dates': dates,
        'values': [float(v) for v in values],
        'message': 'Simple data - install plotly for interactive plots'
    }

@app.route('/')
def index():
    """Main page route"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Template error: {str(e)}. Please ensure templates/index.html exists.", 500

@app.route('/api/data')
def get_data():
    """API endpoint to get data"""
    try:
        if not PLOTLY_AVAILABLE:
            # Return simple data if Plotly is not available
            fallback_data = create_fallback_data()
            return jsonify(fallback_data)
        
        dates, values = generate_sample_data()
        
        if len(dates) == 0 or len(values) == 0:
            return jsonify({'error': 'No data available'}), 500
        
        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name='Sample Data',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title='Sample Time Series Data',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        # Convert to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'graph': graphJSON})
        
    except Exception as e:
        return jsonify({'error': f'Data generation error: {str(e)}'}), 500

@app.route('/plot')
def plot():
    """Route for plot page"""
    try:
        if not PLOTLY_AVAILABLE:
            return "Plotly not installed. Please install with: pip install plotly", 500
        
        dates, values = generate_sample_data()
        
        if len(dates) == 0 or len(values) == 0:
            return "No data available for plotting", 500
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers'))
        fig.update_layout(title='Interactive Plot')
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('plot.html', graphJSON=graphJSON)
    except Exception as e:
        return f"Plot error: {str(e)}. Please ensure templates/plot.html exists.", 500

@app.route('/api/movement-analysis')
def movement_analysis():
    """API endpoint for movement analysis"""
    try:
        # Try to integrate with actual movement analysis system
        try:
            from processing import MovementIdentifier
            
            # Try to analyze actual data
            identifier = MovementIdentifier()
            test_file = "Job Task Data/2025-08-19-21-Reach---1.csv"
            
            if os.path.exists(test_file):
                identifier.load_data(test_file)
                
                # Get basic classifications
                left_movement = identifier.classify_movement_type('left')
                right_movement = identifier.classify_movement_type('right')
                
                # Try to get segmentation data if available
                try:
                    left_segments = identifier.detect_movement_segments('left', min_duration=5)
                    right_segments = identifier.detect_movement_segments('right', min_duration=5)
                    
                    # Detect body movements (bilateral - same for both sides)
                    body_segments = identifier.detect_body_movement_segments(min_duration=5)
                    
                    # Merge body segments into both left and right segments
                    left_segments = sorted(left_segments + body_segments, key=lambda x: x['start_frame'])
                    right_segments = sorted(right_segments + body_segments, key=lambda x: x['start_frame'])
                except:
                    left_segments = []
                    right_segments = []
                
                real_analysis = {
                    'data_source': 'real',
                    'file_analyzed': test_file,
                    'left_hand': {
                        'movement_type': left_movement.value if hasattr(left_movement, 'value') else str(left_movement),
                        'segments': left_segments[:5],  # First 5 segments
                        'total_segments': len(left_segments)
                    },
                    'right_hand': {
                        'movement_type': right_movement.value if hasattr(right_movement, 'value') else str(right_movement),
                        'segments': right_segments[:5],  # First 5 segments  
                        'total_segments': len(right_segments)
                    },
                    'warehouse_operations': ['SHELF_REACHING', 'ITEM_PLACEMENT'],
                    'ergonomic_assessment': 'Real data analysis completed'
                }
                return jsonify(real_analysis)
            
        except ImportError:
            pass  # Fall through to sample data
        except Exception as e:
            print(f"Real analysis failed: {e}")
            
        # Fallback to sample data
        sample_analysis = {
            'data_source': 'sample',
            'message': 'Using sample data - processing.py not available or no data files found',
            'left_hand': {
                'movement_type': 'reach',
                'segments': [
                    {'start_frame': 109, 'end_frame': 117, 'movement_type': 'unknown'},
                    {'start_frame': 769, 'end_frame': 790, 'movement_type': 'reach'},
                    {'start_frame': 883, 'end_frame': 901, 'movement_type': 'push'}
                ],
                'total_segments': 22,
                'coordination_efficiency': 38.6
            },
            'right_hand': {
                'movement_type': 'push', 
                'segments': [
                    {'start_frame': 108, 'end_frame': 117, 'movement_type': 'unknown'},
                    {'start_frame': 253, 'end_frame': 277, 'movement_type': 'push'},
                    {'start_frame': 401, 'end_frame': 447, 'movement_type': 'push'}
                ],
                'total_segments': 17,
                'coordination_efficiency': 38.6
            },
            'warehouse_operations': ['SHELF_REACHING', 'ITEM_PLACEMENT'],
            'ergonomic_assessment': 'MODERATE COORDINATION - Assembly work'
        }
        return jsonify(sample_analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/movement-plot')
def movement_plot():
    """Generate plot data for movement analysis"""
    try:
        if not PLOTLY_AVAILABLE:
            return jsonify({'error': 'Plotly not available for movement plots'}), 500
            
        # Try to get real movement data
        try:
            from processing import MovementIdentifier
            
            identifier = MovementIdentifier()
            test_file = "Job Task Data/2025-08-19-21-Reach---1.csv" 
            
            if os.path.exists(test_file):
                identifier.load_data(test_file)
                
                # Get wrist coordinates
                left_wrist = identifier.get_joint_coordinates('wrist', 'left')
                right_wrist = identifier.get_joint_coordinates('wrist', 'right')
                
                if len(left_wrist) > 0 and len(right_wrist) > 0:
                    # Create frame numbers
                    frames = list(range(len(left_wrist)))
                    
                    # Create multi-trace plot
                    fig = go.Figure()
                    
                    # Add left wrist trajectory
                    fig.add_trace(go.Scatter(
                        x=frames,
                        y=[pos[0] for pos in left_wrist],  # X coordinate
                        mode='lines',
                        name='Left Wrist X',
                        line=dict(color='blue', width=1)
                    ))
                    
                    # Add right wrist trajectory  
                    fig.add_trace(go.Scatter(
                        x=frames,
                        y=[pos[0] for pos in right_wrist],  # X coordinate
                        mode='lines',
                        name='Right Wrist X',
                        line=dict(color='red', width=1)
                    ))
                    
                    fig.update_layout(
                        title='Hand Movement Analysis - Wrist X Coordinates',
                        xaxis_title='Frame Number',
                        yaxis_title='X Position (meters)',
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                    return jsonify({'graph': graphJSON, 'source': 'real_data'})
        except:
            pass  # Fall through to sample data
            
        # Fallback: create sample movement data
        frames = list(range(100))
        
        # Simulate hand movements
        left_x = [0.5 + 0.1 * np.sin(i * 0.1) + 0.05 * np.random.randn() for i in frames]
        right_x = [0.3 + 0.15 * np.cos(i * 0.08) + 0.05 * np.random.randn() for i in frames]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=frames,
            y=left_x,
            mode='lines+markers',
            name='Left Hand Movement',
            line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=frames,
            y=right_x,
            mode='lines+markers', 
            name='Right Hand Movement',
            line=dict(color='red', width=1)
        ))
        
        fig.update_layout(
            title='Simulated Hand Movement Analysis',
            xaxis_title='Frame Number',
            yaxis_title='Position (meters)',
            hovermode='x unified'
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'graph': graphJSON, 'source': 'simulated'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload for movement data analysis"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        # Validate file
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Add timestamp to avoid conflicts
            import time
            timestamp = str(int(time.time()))
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            
            # Save file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Validate CSV structure
            is_valid, message = validate_csv_structure(filepath)
            
            if not is_valid:
                os.remove(filepath)  # Clean up invalid file
                flash(f'Invalid file: {message}')
                return redirect(request.url)
            
            flash(f'File uploaded successfully: {file.filename}')
            return redirect(url_for('analyze_uploaded_file', filename=filename))
        else:
            flash('Invalid file type. Please upload a CSV file.')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """API endpoint for file upload via AJAX"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        import time
        timestamp = str(int(time.time()))
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        
        # Save file using appropriate storage method
        if not save_uploaded_file(file, filename):
            return jsonify({'error': 'Failed to save file'}), 500
        
        # Get file path for validation
        filepath = get_file_path(filename)
        if not filepath:
            return jsonify({'error': 'Failed to retrieve file'}), 500
        
        # Validate structure
        is_valid, message = validate_csv_structure(filepath)
        
        if not is_valid:
            os.remove(filepath)
            return jsonify({'error': message}), 400
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'File uploaded successfully',
            'analyze_url': url_for('analyze_uploaded_file', filename=filename)
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/uploaded-files')
def uploaded_files():
    """View uploaded files and analyze them"""
    files = []
    filenames = list_uploaded_files()
    
    for filename in filenames:
        if filename.endswith('.csv'):
            filepath = get_file_path(filename)
            if filepath and os.path.exists(filepath):
                file_info = {
                    'name': filename,
                    'size': os.path.getsize(filepath),
                    'modified': os.path.getmtime(filepath)
                }
                files.append(file_info)
    
    if not files:
        return render_template('upload.html', error="No uploaded files found")
    
    return render_template('uploaded_files.html', files=files)

@app.route('/analyze-uploaded/<filename>')
def analyze_uploaded_file(filename):
    """Analyze a specific uploaded file"""
    try:
        # Get file path using storage helper
        filepath = get_file_path(filename)
        
        if not filepath or not os.path.exists(filepath):
            logger.error(f"File not found: {filename}")
            return jsonify({'error': 'File not found'}), 404
            
        if not PROCESSING_AVAILABLE:
            return jsonify({'error': 'Processing module not available'}), 500
            
        # Import and use the processing module
        from processing import MovementIdentifier
        
        identifier = MovementIdentifier()
        identifier.load_data(filepath)
        
        # Get movement segments for both sides
        left_segments = identifier.detect_movement_segments('left', min_duration=5)
        right_segments = identifier.detect_movement_segments('right', min_duration=5)
        
        # Detect body movements (bilateral - same for both sides)
        body_segments = identifier.detect_body_movement_segments(min_duration=5)
        
        # Merge body segments into both left and right segments
        left_segments = sorted(left_segments + body_segments, key=lambda x: x['start_frame'])
        right_segments = sorted(right_segments + body_segments, key=lambda x: x['start_frame'])
        
        # Get coordinate data for plotting
        left_wrist = identifier.get_joint_coordinates('wrist', 'left')
        right_wrist = identifier.get_joint_coordinates('wrist', 'right')
        
        # Convert to list if numpy array and check for empty data
        if hasattr(left_wrist, '__len__') and hasattr(right_wrist, '__len__'):
            left_len = len(left_wrist) if left_wrist is not None else 0
            right_len = len(right_wrist) if right_wrist is not None else 0
            if left_len == 0 and right_len == 0:
                return jsonify({'error': 'No movement data found in file'}), 400
        else:
            return jsonify({'error': 'Invalid movement data format'}), 400
            
        
        if PLOTLY_AVAILABLE:
            # Determine frame count safely
            left_len = len(left_wrist) if left_wrist is not None and hasattr(left_wrist, '__len__') else 0
            right_len = len(right_wrist) if right_wrist is not None and hasattr(right_wrist, '__len__') else 0
            frame_count = max(left_len, right_len)
            frames = list(range(frame_count))
            
            # Get shoulder coordinates for relative positions
            left_shoulder = identifier.get_joint_coordinates('shoulder', 'left')
            right_shoulder = identifier.get_joint_coordinates('shoulder', 'right')
            
            # Calculate horizontal positions relative to shoulders (X-axis)
            left_wrist_x_rel = []
            right_wrist_x_rel = []
            
            if left_wrist is not None and left_len > 0 and left_shoulder is not None:
                for i in range(min(left_len, len(left_shoulder))):
                    left_wrist_x_rel.append(left_wrist[i][0] - left_shoulder[i][0])
            
            if right_wrist is not None and right_len > 0 and right_shoulder is not None:
                for i in range(min(right_len, len(right_shoulder))):
                    right_wrist_x_rel.append(right_wrist[i][0] - right_shoulder[i][0])
            
            # Calculate vertical positions relative to shoulders (Z-axis)
            left_wrist_z_rel = []
            right_wrist_z_rel = []
            
            if left_wrist is not None and left_len > 0 and left_shoulder is not None:
                for i in range(min(left_len, len(left_shoulder))):
                    left_wrist_z_rel.append(left_wrist[i][2] - left_shoulder[i][2])
            
            if right_wrist is not None and right_len > 0 and right_shoulder is not None:
                for i in range(min(right_len, len(right_shoulder))):
                    right_wrist_z_rel.append(right_wrist[i][2] - right_shoulder[i][2])
            
            # Create horizontal position plot
            fig_horizontal = go.Figure()
            
            if len(left_wrist_x_rel) > 0:
                fig_horizontal.add_trace(go.Scatter(
                    x=frames[:len(left_wrist_x_rel)],
                    y=left_wrist_x_rel,
                    mode='lines',
                    name='Left Wrist',
                    line=dict(color='blue', width=2)
                ))
            
            if len(right_wrist_x_rel) > 0:
                fig_horizontal.add_trace(go.Scatter(
                    x=frames[:len(right_wrist_x_rel)],
                    y=right_wrist_x_rel,
                    mode='lines', 
                    name='Right Wrist',
                    line=dict(color='red', width=2)
                ))
            
            fig_horizontal.update_layout(
                title='Horizontal Position Relative to Shoulder (X-axis)',
                xaxis_title='Frame Number',
                yaxis_title='Horizontal Position (meters)',
                hovermode='x unified',
                height=400
            )
            
            # Create vertical position plot
            fig_vertical = go.Figure()
            
            if len(left_wrist_z_rel) > 0:
                fig_vertical.add_trace(go.Scatter(
                    x=frames[:len(left_wrist_z_rel)],
                    y=left_wrist_z_rel,
                    mode='lines',
                    name='Left Wrist',
                    line=dict(color='blue', width=2)
                ))
            
            if len(right_wrist_z_rel) > 0:
                fig_vertical.add_trace(go.Scatter(
                    x=frames[:len(right_wrist_z_rel)],
                    y=right_wrist_z_rel,
                    mode='lines', 
                    name='Right Wrist',
                    line=dict(color='red', width=2)
                ))
            
            fig_vertical.update_layout(
                title='Vertical Position Relative to Shoulder (Z-axis)',
                xaxis_title='Frame Number',
                yaxis_title='Vertical Position (meters)',
                hovermode='x unified',
                height=400
            )
            
            graphJSON_horizontal = json.dumps(fig_horizontal, cls=plotly.utils.PlotlyJSONEncoder)
            graphJSON_vertical = json.dumps(fig_vertical, cls=plotly.utils.PlotlyJSONEncoder)
        else:
            graphJSON_horizontal = None
            graphJSON_vertical = None
            
        return render_template('analysis_results.html', 
                             filename=filename,
                             left_segments=left_segments,
                             right_segments=right_segments,
                             graphJSON_horizontal=graphJSON_horizontal,
                             graphJSON_vertical=graphJSON_vertical,
                             total_frames=frame_count if 'frame_count' in locals() else 0)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/download-report/<filename>')
def download_report(filename):
    """Generate and download analysis report as text file"""
    try:
        from flask import send_file, make_response
        from io import BytesIO
        import datetime
        
        # Get file path using storage helper
        filepath = get_file_path(filename)
        
        if not filepath or not os.path.exists(filepath):
            logger.error(f"File not found for report: {filename}")
            return jsonify({'error': 'File not found'}), 404
            
        if not PROCESSING_AVAILABLE:
            return jsonify({'error': 'Processing module not available'}), 500
            
        # Import and use the processing module
        from processing import MovementIdentifier
        
        identifier = MovementIdentifier()
        identifier.load_data(filepath)
        
        # Analyze both hands
        left_segments = identifier.detect_movement_segments('left', min_duration=5)
        right_segments = identifier.detect_movement_segments('right', min_duration=5)
        
        # Detect body movements (bilateral - same for both sides)
        body_segments = identifier.detect_body_movement_segments(min_duration=5)
        
        # Merge body segments into both left and right segments
        # Body movements are bilateral so they appear in both reports
        left_segments = sorted(left_segments + body_segments, key=lambda x: x['start_frame'])
        right_segments = sorted(right_segments + body_segments, key=lambda x: x['start_frame'])
        
        # Calculate joint angles for biomechanical analysis
        left_angles = identifier.calculate_joint_angles('left')
        right_angles = identifier.calculate_joint_angles('right')
        
        # Get joint coordinates for position analysis
        left_shoulder_coords = identifier.get_joint_coordinates('shoulder', 'left')
        left_elbow_coords = identifier.get_joint_coordinates('elbow', 'left')
        left_wrist_coords = identifier.get_joint_coordinates('wrist', 'left')
        
        right_shoulder_coords = identifier.get_joint_coordinates('shoulder', 'right')
        right_elbow_coords = identifier.get_joint_coordinates('elbow', 'right')
        right_wrist_coords = identifier.get_joint_coordinates('wrist', 'right')
        
        # Calculate horizontal positions relative to shoulder (X-axis, index 0)
        # Positive values = right of shoulder, Negative = left of shoulder
        left_elbow_x_rel = left_elbow_coords[:, 0] - left_shoulder_coords[:, 0]
        left_wrist_x_rel = left_wrist_coords[:, 0] - left_shoulder_coords[:, 0]
        
        right_elbow_x_rel = right_elbow_coords[:, 0] - right_shoulder_coords[:, 0]
        right_wrist_x_rel = right_wrist_coords[:, 0] - right_shoulder_coords[:, 0]
        
        # Calculate vertical positions relative to shoulder (Z-axis, index 2)
        # Positive values = above shoulder, Negative = below shoulder
        left_elbow_z_rel = left_elbow_coords[:, 2] - left_shoulder_coords[:, 2]
        left_wrist_z_rel = left_wrist_coords[:, 2] - left_shoulder_coords[:, 2]
        
        right_elbow_z_rel = right_elbow_coords[:, 2] - right_shoulder_coords[:, 2]
        right_wrist_z_rel = right_wrist_coords[:, 2] - right_shoulder_coords[:, 2]
        
        # Extract shoulder angle statistics
        left_shoulder_angles = left_angles.get('left_shoulder_angle', np.array([]))
        right_shoulder_angles = right_angles.get('right_shoulder_angle', np.array([]))
        
        left_shoulder_max = np.max(left_shoulder_angles) if len(left_shoulder_angles) > 0 else 0
        left_shoulder_mean = np.mean(left_shoulder_angles) if len(left_shoulder_angles) > 0 else 0
        right_shoulder_max = np.max(right_shoulder_angles) if len(right_shoulder_angles) > 0 else 0
        right_shoulder_mean = np.mean(right_shoulder_angles) if len(right_shoulder_angles) > 0 else 0
        
        # Get overall classifications
        try:
            left_movement = identifier.classify_movement_type('left')
            right_movement = identifier.classify_movement_type('right')
            left_movement_str = left_movement.value if hasattr(left_movement, 'value') else str(left_movement)
            right_movement_str = right_movement.value if hasattr(right_movement, 'value') else str(right_movement)
        except:
            left_movement_str = 'unknown'
            right_movement_str = 'unknown'
        
        # Generate report content
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MOVEMENT ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Source File: {filename}")
        report_lines.append(f"Total Frames: {len(identifier.data) if identifier.data is not None else 0}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall classifications
        report_lines.append("OVERALL MOVEMENT CLASSIFICATION")
        report_lines.append("-" * 80)
        report_lines.append(f"Left Hand:  {left_movement_str.upper()}")
        report_lines.append(f"Right Hand: {right_movement_str.upper()}")
        report_lines.append("")
        
        # Biomechanical metrics
        report_lines.append("BIOMECHANICAL METRICS - SHOULDER ANGLES")
        report_lines.append("-" * 80)
        report_lines.append(f"Left Shoulder:")
        report_lines.append(f"  Maximum Angle:  {left_shoulder_max:>6.1f}°")
        report_lines.append(f"  Mean Angle:     {left_shoulder_mean:>6.1f}°")
        report_lines.append("")
        report_lines.append(f"Right Shoulder:")
        report_lines.append(f"  Maximum Angle:  {right_shoulder_max:>6.1f}°")
        report_lines.append(f"  Mean Angle:     {right_shoulder_mean:>6.1f}°")
        report_lines.append("")
        
        # Left hand segments
        report_lines.append("LEFT HAND MOVEMENT SEGMENTS")
        report_lines.append("=" * 120)
        report_lines.append(f"Total Segments: {len(left_segments)}")
        report_lines.append("")
        
        if left_segments:
            # Header with all metrics
            report_lines.append(f"Seg\tStart\tEnd\tDur\tType\t\t\tSh.Ang\tSh.Ang\tWrs.X\tWrs.X\tElb.Z\tElb.Z\tWrs.Z\tWrs.Z")
            report_lines.append(f"#\tFrame\tFrame\t(s)\t\t\t\tMax(°)\tMean(°)\tMax(m)\tMean(m)\tMax(m)\tMean(m)\tMax(m)\tMean(m)")
            report_lines.append("-" * 140)
            
            for i, seg in enumerate(left_segments, 1):
                movement_type = seg.get('movement_type', 'unknown').upper()
                start = seg.get('start_frame', 0)
                end = seg.get('end_frame', 0)
                duration = seg.get('duration_seconds', 0.0)
                
                # Add extra tabs based on movement type length for alignment
                if len(movement_type) <= 7:  # Short names like REACH, PULL, PUSH, LIFT, CARRY, WALK
                    type_tabs = "\t\t"
                elif len(movement_type) <= 15:  # Medium names like SQUAT, UNKNOWN
                    type_tabs = "\t"
                else:  # Long names like LOWER_BACK_BEND
                    type_tabs = ""
                
                # Calculate shoulder angles for this segment
                if len(left_shoulder_angles) > 0 and end <= len(left_shoulder_angles):
                    seg_shoulder_angles = left_shoulder_angles[start:end+1]
                    seg_shoulder_max = np.max(seg_shoulder_angles) if len(seg_shoulder_angles) > 0 else 0
                    seg_shoulder_mean = np.mean(seg_shoulder_angles) if len(seg_shoulder_angles) > 0 else 0
                else:
                    seg_shoulder_max = 0
                    seg_shoulder_mean = 0
                
                # Calculate wrist horizontal positions relative to shoulder for this segment
                if len(left_wrist_x_rel) > 0 and end <= len(left_wrist_x_rel):
                    seg_wrist_x = left_wrist_x_rel[start:end+1]
                    seg_wrist_x_max = np.max(seg_wrist_x) if len(seg_wrist_x) > 0 else 0
                    seg_wrist_x_mean = np.mean(seg_wrist_x) if len(seg_wrist_x) > 0 else 0
                else:
                    seg_wrist_x_max = 0
                    seg_wrist_x_mean = 0
                
                # Calculate elbow vertical position for this segment
                if len(left_elbow_z_rel) > 0 and end <= len(left_elbow_z_rel):
                    seg_elbow_z = left_elbow_z_rel[start:end+1]
                    seg_elbow_z_max = np.max(seg_elbow_z) if len(seg_elbow_z) > 0 else 0
                    seg_elbow_z_mean = np.mean(seg_elbow_z) if len(seg_elbow_z) > 0 else 0
                else:
                    seg_elbow_z_max = 0
                    seg_elbow_z_mean = 0
                
                # Calculate wrist vertical position for this segment
                if len(left_wrist_z_rel) > 0 and end <= len(left_wrist_z_rel):
                    seg_wrist_z = left_wrist_z_rel[start:end+1]
                    seg_wrist_z_max = np.max(seg_wrist_z) if len(seg_wrist_z) > 0 else 0
                    seg_wrist_z_mean = np.mean(seg_wrist_z) if len(seg_wrist_z) > 0 else 0
                else:
                    seg_wrist_z_max = 0
                    seg_wrist_z_mean = 0
                
                report_lines.append(f"{i}\t{start}\t{end}\t{duration:.2f}\t{movement_type}{type_tabs}\t"
                                  f"{seg_shoulder_max:.1f}\t{seg_shoulder_mean:.1f}\t"
                                  f"{seg_wrist_x_max:.3f}\t{seg_wrist_x_mean:.3f}\t"
                                  f"{seg_elbow_z_max:.3f}\t{seg_elbow_z_mean:.3f}\t"
                                  f"{seg_wrist_z_max:.3f}\t{seg_wrist_z_mean:.3f}")
        else:
            report_lines.append("No movement segments detected")
        
        report_lines.append("")
        report_lines.append("Note: All positions (X, Z) are relative to corresponding shoulder")
        report_lines.append("")
        
        # Right hand segments
        report_lines.append("RIGHT HAND MOVEMENT SEGMENTS")
        report_lines.append("=" * 120)
        report_lines.append(f"Total Segments: {len(right_segments)}")
        report_lines.append("")
        
        if right_segments:
            # Header with all metrics
            report_lines.append(f"Seg\tStart\tEnd\tDur\tType\t\t\tSh.Ang\tSh.Ang\tWrs.X\tWrs.X\tElb.Z\tElb.Z\tWrs.Z\tWrs.Z")
            report_lines.append(f"#\tFrame\tFrame\t(s)\t\t\t\tMax(°)\tMean(°)\tMax(m)\tMean(m)\tMax(m)\tMean(m)\tMax(m)\tMean(m)")
            report_lines.append("-" * 140)
            
            for i, seg in enumerate(right_segments, 1):
                movement_type = seg.get('movement_type', 'unknown').upper()
                start = seg.get('start_frame', 0)
                end = seg.get('end_frame', 0)
                duration = seg.get('duration_seconds', 0.0)
                
                # Add extra tabs based on movement type length for alignment
                if len(movement_type) <= 7:  # Short names like REACH, PULL, PUSH, LIFT, CARRY, WALK
                    type_tabs = "\t\t"
                elif len(movement_type) <= 15:  # Medium names like SQUAT, UNKNOWN
                    type_tabs = "\t"
                else:  # Long names like LOWER_BACK_BEND
                    type_tabs = ""
                
                # Calculate shoulder angles for this segment
                if len(right_shoulder_angles) > 0 and end <= len(right_shoulder_angles):
                    seg_shoulder_angles = right_shoulder_angles[start:end+1]
                    seg_shoulder_max = np.max(seg_shoulder_angles) if len(seg_shoulder_angles) > 0 else 0
                    seg_shoulder_mean = np.mean(seg_shoulder_angles) if len(seg_shoulder_angles) > 0 else 0
                else:
                    seg_shoulder_max = 0
                    seg_shoulder_mean = 0
                
                # Calculate wrist horizontal positions relative to shoulder for this segment
                if len(right_wrist_x_rel) > 0 and end <= len(right_wrist_x_rel):
                    seg_wrist_x = right_wrist_x_rel[start:end+1]
                    seg_wrist_x_max = np.max(seg_wrist_x) if len(seg_wrist_x) > 0 else 0
                    seg_wrist_x_mean = np.mean(seg_wrist_x) if len(seg_wrist_x) > 0 else 0
                else:
                    seg_wrist_x_max = 0
                    seg_wrist_x_mean = 0
                
                # Calculate elbow vertical position for this segment
                if len(right_elbow_z_rel) > 0 and end <= len(right_elbow_z_rel):
                    seg_elbow_z = right_elbow_z_rel[start:end+1]
                    seg_elbow_z_max = np.max(seg_elbow_z) if len(seg_elbow_z) > 0 else 0
                    seg_elbow_z_mean = np.mean(seg_elbow_z) if len(seg_elbow_z) > 0 else 0
                else:
                    seg_elbow_z_max = 0
                    seg_elbow_z_mean = 0
                
                # Calculate wrist vertical position for this segment
                if len(right_wrist_z_rel) > 0 and end <= len(right_wrist_z_rel):
                    seg_wrist_z = right_wrist_z_rel[start:end+1]
                    seg_wrist_z_max = np.max(seg_wrist_z) if len(seg_wrist_z) > 0 else 0
                # Calculate wrist vertical position for this segment
                if len(right_wrist_z_rel) > 0 and end <= len(right_wrist_z_rel):
                    seg_wrist_z = right_wrist_z_rel[start:end+1]
                    seg_wrist_z_max = np.max(seg_wrist_z) if len(seg_wrist_z) > 0 else 0
                    seg_wrist_z_mean = np.mean(seg_wrist_z) if len(seg_wrist_z) > 0 else 0
                else:
                    seg_wrist_z_max = 0
                    seg_wrist_z_mean = 0
                
                report_lines.append(f"{i}\t{start}\t{end}\t{duration:.2f}\t{movement_type}{type_tabs}\t"
                                  f"{seg_shoulder_max:.1f}\t{seg_shoulder_mean:.1f}\t"
                                  f"{seg_wrist_x_max:.3f}\t{seg_wrist_x_mean:.3f}\t"
                                  f"{seg_elbow_z_max:.3f}\t{seg_elbow_z_mean:.3f}\t"
                                  f"{seg_wrist_z_max:.3f}\t{seg_wrist_z_mean:.3f}")
        else:
            report_lines.append("No movement segments detected")
        
        report_lines.append("")
        report_lines.append("METRIC DEFINITIONS:")
        report_lines.append("-" * 140)
        report_lines.append("Sh.Ang  = Shoulder angle (upper_spine-shoulder-elbow), in degrees")
        report_lines.append("Wrs.X   = Wrist horizontal position relative to shoulder (X-axis, same as Horizontal plot)")
        report_lines.append("            Positive = right of shoulder, Negative = left of shoulder, in meters")
        report_lines.append("Elb.Z   = Elbow vertical position relative to shoulder (Z-axis, same as Vertical plot)")
        report_lines.append("            Positive = above shoulder, Negative = below shoulder, in meters")
        report_lines.append("Wrs.Z   = Wrist vertical position relative to shoulder (Z-axis, same as Vertical plot)")
        report_lines.append("            Positive = above shoulder, Negative = below shoulder, in meters")
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 80)
        
        if left_segments:
            left_durations = [seg.get('duration_seconds', 0) for seg in left_segments]
            left_avg_duration = sum(left_durations) / len(left_durations) if left_durations else 0
            report_lines.append(f"Left Hand Average Segment Duration:  {left_avg_duration:.2f} seconds")
        
        if right_segments:
            right_durations = [seg.get('duration_seconds', 0) for seg in right_segments]
            right_avg_duration = sum(right_durations) / len(right_durations) if right_durations else 0
            report_lines.append(f"Right Hand Average Segment Duration: {right_avg_duration:.2f} seconds")
        
        # Movement type distribution
        report_lines.append("")
        report_lines.append("MOVEMENT TYPE DISTRIBUTION")
        report_lines.append("-" * 80)
        
        # Count movement types for left hand
        if left_segments:
            left_types = {}
            for seg in left_segments:
                mv_type = seg.get('movement_type', 'unknown')
                left_types[mv_type] = left_types.get(mv_type, 0) + 1
            
            report_lines.append("Left Hand:")
            for mv_type, count in sorted(left_types.items()):
                percentage = (count / len(left_segments)) * 100
                report_lines.append(f"  {mv_type.upper():<15} {count:>3} segments ({percentage:>5.1f}%)")
        
        report_lines.append("")
        
        # Count movement types for right hand
        if right_segments:
            right_types = {}
            for seg in right_segments:
                mv_type = seg.get('movement_type', 'unknown')
                right_types[mv_type] = right_types.get(mv_type, 0) + 1
            
            report_lines.append("Right Hand:")
            for mv_type, count in sorted(right_types.items()):
                percentage = (count / len(right_segments)) * 100
                report_lines.append(f"  {mv_type.upper():<15} {count:>3} segments ({percentage:>5.1f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Join all lines into report text
        report_text = "\n".join(report_lines)
        
        # Create BytesIO object
        buffer = BytesIO()
        buffer.write(report_text.encode('utf-8'))
        buffer.seek(0)
        
        # Create response
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=movement_analysis_report_{filename[:-4]}.txt'
        
        return response
        
    except Exception as e:
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Server status and diagnostics"""
    template_dir = os.path.join(os.getcwd(), 'templates')
    templates_exist = os.path.exists(template_dir)
    
    if templates_exist:
        template_files = os.listdir(template_dir)
    else:
        template_files = []
    
    status_info = {
        'server': 'running',
        'template_directory': template_dir,
        'templates_exist': templates_exist,
        'template_files': template_files,
        'current_directory': os.getcwd()
    }
    return jsonify(status_info)

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Template directory: {os.path.join(os.getcwd(), 'templates')}")
    logger.info(f"Upload directory: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Production mode: {IS_PRODUCTION}")
    
    # Check if templates exist
    if os.path.exists('templates'):
        logger.info("✓ Templates directory found")
        templates = os.listdir('templates')
        logger.info(f"✓ Template files: {templates}")
    else:
        logger.warning("✗ Templates directory not found - creating it...")
        os.makedirs('templates')
    
    # Run with appropriate settings
    if IS_PRODUCTION:
        # Production: gunicorn will handle this
        logger.info("Running in production mode (gunicorn)")
    else:
        # Development: use built-in server
        logger.info("Running in development mode")
        app.run(debug=True, host='127.0.0.1', port=5000)