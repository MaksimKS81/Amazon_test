from flask import Flask, jsonify, render_template_string

app = Flask(__name__)


@app.route('/')
def home():
    """Home page with basic HTML"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flask Test App</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
            }
            .success {
                color: #28a745;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎉 Flask Test Application</h1>
            <p class="success">✓ Application is running successfully!</p>
            <p>This Flask app is ready for Google Cloud deployment.</p>
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/api/health">Health Check API</a></li>
                <li><a href="/api/test">Test API</a></li>
            </ul>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Application is running'
    })


@app.route('/api/test')
def test_endpoint():
    """Test API endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Test endpoint working',
        'data': {
            'version': '1.0.0',
            'environment': 'production'
        }
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'Something went wrong'
    }), 500


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google Cloud,
    # a production-grade WSGI server (gunicorn) is used instead.
    app.run(host='0.0.0.0', port=8080, debug=True)
