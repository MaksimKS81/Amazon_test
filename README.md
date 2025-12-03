# Flask Test Project for Google Cloud

A simple Flask web application configured for deployment to Google Cloud Platform (App Engine).

## Features

- ✅ Basic Flask application with multiple endpoints
- ✅ Health check API endpoint
- ✅ Test API endpoint with JSON responses
- ✅ Error handling (404, 500)
- ✅ Ready for Google Cloud App Engine deployment
- ✅ Production-ready with gunicorn

## Project Structure

```
Amazon_test/
├── main.py              # Flask application
├── requirements.txt     # Python dependencies
├── app.yaml            # Google Cloud App Engine configuration
├── .gcloudignore       # Files to exclude from deployment
└── README.md           # This file
```

## Local Development

### Prerequisites

- Python 3.12 or higher
- pip (Python package installer)

### Setup and Run Locally

1. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```powershell
   python main.py
   ```

4. **Access the application**:
   - Open your browser and go to: `http://localhost:8080`
   - Health check: `http://localhost:8080/api/health`
   - Test API: `http://localhost:8080/api/test`

## Available Endpoints

- `GET /` - Home page with HTML interface
- `GET /api/health` - Health check endpoint (returns JSON)
- `GET /api/test` - Test endpoint with sample data (returns JSON)

## Google Cloud Deployment

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
- A Google Cloud project created
- Billing enabled on your Google Cloud project

### Deploy to Google Cloud App Engine

1. **Initialize gcloud (first time only)**:
   ```powershell
   gcloud init
   ```

2. **Set your project** (replace PROJECT_ID with your actual project ID):
   ```powershell
   gcloud config set project PROJECT_ID
   ```

3. **Deploy the application**:
   ```powershell
   gcloud app deploy
   ```

4. **Open the deployed application**:
   ```powershell
   gcloud app browse
   ```

### View Logs

```powershell
gcloud app logs tail -s default
```

### Check Application Status

```powershell
gcloud app describe
```

## Configuration

### app.yaml

The `app.yaml` file configures the App Engine environment:
- **Runtime**: Python 3.12
- **Instance class**: F1 (free tier eligible)
- **Scaling**: Automatic (1-10 instances)
- **Entrypoint**: Gunicorn WSGI server

You can modify these settings based on your needs.

## Testing the Deployed Application

After deployment, test your endpoints:

```powershell
# Get your app URL
$APP_URL = gcloud app describe --format="value(defaultHostname)"

# Test home page
curl "https://$APP_URL/"

# Test health endpoint
curl "https://$APP_URL/api/health"

# Test API endpoint
curl "https://$APP_URL/api/test"
```

## Troubleshooting

### Common Issues

1. **"File not found" error after upload on Google Cloud**
   - **Cause**: Google Cloud App Engine uses ephemeral storage. Files uploaded to `/tmp` are lost between requests
   - **Solution**: The app is configured to use Google Cloud Storage in production
   - **Required**: Set up a GCS bucket:
     ```powershell
     # Create a bucket (must be globally unique name)
     gsutil mb gs://your-unique-bucket-name
     
     # Set environment variable in app.yaml
     # Add to env_variables section:
     # GCS_BUCKET_NAME: "your-unique-bucket-name"
     ```
   - **Alternative**: For testing only, use Cloud Storage FUSE or implement file upload differently

2. **"ERROR: (gcloud.app.deploy) NOT_FOUND: Unable to retrieve P4SA"**
   - Run: `gcloud app create` to initialize App Engine in your project

3. **Permission denied errors**
   - Ensure you have the necessary IAM roles (App Engine Admin, Service Account User)
   - Grant App Engine default service account access to GCS bucket:
     ```powershell
     gsutil iam ch serviceAccount:YOUR-PROJECT@appspot.gserviceaccount.com:objectAdmin gs://your-bucket-name
     ```

4. **Port binding issues locally**
   - The app uses port 8080. Make sure it's not in use by another application
   - For local development, `run_local.py` uses port 5000 instead

5. **Missing dependencies**
   - Ensure all packages in `requirements.txt` are installed:
     ```powershell
     pip install -r requirements.txt
     ```

6. **Template not found errors**
   - Ensure `templates/` directory exists with all HTML files
   - Check that templates are not listed in `.gcloudignore`

### View Error Logs

```powershell
gcloud app logs read --service=default --limit=50
```

## Cost Considerations

- App Engine offers a free tier with quotas
- F1 instance class is eligible for free tier
- Monitor usage in Google Cloud Console
- Consider setting up billing alerts

## License

This is a test project for demonstration purposes.