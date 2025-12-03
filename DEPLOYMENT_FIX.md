# DEPLOYMENT FIX - File Upload Issue

## Problem
After deploying to Google Cloud, CSV file uploads result in "File not found" error when trying to analyze.

## Root Cause
Google App Engine uses ephemeral filesystem - files saved to local disk are lost between requests. The `/tmp` directory is only persistent within a single request.

## Solution Implemented

### 1. Fixed File Path Handling (✓ DONE)
Updated `flask_server.py` to use storage helper functions consistently:
- `save_uploaded_file()` - Handles both local and GCS storage
- `get_file_path()` - Retrieves file from appropriate storage
- `list_uploaded_files()` - Lists files from local or GCS

### 2. Google Cloud Storage Integration (⚠ NEEDS CONFIGURATION)
The app detects production environment and uses GCS automatically, but requires setup:

**Required Steps:**

1. **Create a GCS Bucket**
   ```powershell
   # Replace with your unique bucket name
   gsutil mb gs://movement-analysis-uploads-YOUR-PROJECT-ID
   ```

2. **Update app.yaml**
   Add the bucket name to environment variables:
   ```yaml
   env_variables:
     PYTHONUNBUFFERED: "1"
     GCS_BUCKET_NAME: "movement-analysis-uploads-YOUR-PROJECT-ID"
   ```

3. **Set Bucket Permissions**
   ```powershell
   # Get your project ID
   $PROJECT_ID = gcloud config get-value project
   
   # Grant access to App Engine service account
   gsutil iam ch serviceAccount:${PROJECT_ID}@appspot.gserviceaccount.com:objectAdmin gs://movement-analysis-uploads-${PROJECT_ID}
   ```

4. **Deploy with new configuration**
   ```powershell
   gcloud app deploy
   ```

## Testing

### Local Testing
```powershell
# Run locally (uses local 'uploads' folder)
python run_local.py
```

### Production Testing
1. Deploy to Google Cloud
2. Upload a CSV file via the web interface
3. Click "Analyze" button
4. File should now be retrieved from GCS successfully

## How It Works

**Local Development:**
- Files saved to: `./uploads/` directory
- Files retrieved from: `./uploads/` directory

**Production (Google Cloud):**
- Files saved to: Google Cloud Storage bucket
- Files retrieved from: GCS bucket (downloaded to `/tmp/` temporarily for analysis)
- Original file persists in GCS even after request completes

## Alternative Solutions

If you don't want to use GCS:

1. **Use Cloud Storage FUSE** - Mount GCS bucket as filesystem
2. **Use Cloud SQL or Firestore** - Store file metadata and content
3. **Process immediately** - Analyze file during upload request (before it's lost)

## Files Modified
- ✓ `flask_server.py` - Fixed file path handling in 4 locations
- ✓ `.gcloudignore` - Added uploads/ to exclusion list
- ✓ `README.md` - Added troubleshooting section
- ✓ `run_local.py` - Created local development helper

## Next Deployment Steps
1. Configure GCS bucket (see above)
2. Update `app.yaml` with bucket name
3. Set bucket permissions
4. Deploy: `gcloud app deploy`
5. Test upload and analysis functionality
