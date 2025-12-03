# Google Cloud Storage Setup Script
# Run this after creating your GCP project

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "GCS Bucket Setup for File Uploads" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# Get current project
$PROJECT_ID = gcloud config get-value project 2>$null

if (-not $PROJECT_ID) {
    Write-Host "Error: No GCP project set. Run 'gcloud init' first." -ForegroundColor Red
    exit 1
}

Write-Host "Current GCP Project: $PROJECT_ID" -ForegroundColor Green
Write-Host ""

# Generate bucket name
$BUCKET_NAME = "movement-analysis-uploads-$PROJECT_ID"

Write-Host "Creating GCS bucket: $BUCKET_NAME" -ForegroundColor Yellow

# Create bucket
gsutil mb -p $PROJECT_ID gs://$BUCKET_NAME

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Bucket created successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to create bucket (may already exist)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setting bucket permissions..." -ForegroundColor Yellow

# Set permissions for App Engine service account
$SERVICE_ACCOUNT = "${PROJECT_ID}@appspot.gserviceaccount.com"
gsutil iam ch serviceAccount:${SERVICE_ACCOUNT}:objectAdmin gs://$BUCKET_NAME

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Permissions set successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to set permissions" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Update app.yaml with the following:" -ForegroundColor White
Write-Host ""
Write-Host "   env_variables:" -ForegroundColor Gray
Write-Host "     GCS_BUCKET_NAME: `"$BUCKET_NAME`"" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Deploy your application:" -ForegroundColor White
Write-Host "   gcloud app deploy" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Test file upload functionality" -ForegroundColor White
Write-Host ""
