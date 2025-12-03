# Session-Based File Storage Using /tmp

## Summary

All uploaded files are now stored in `/tmp` directory for the duration of the session.

## How It Works

### Local Development
- Files stored in: `/tmp/`
- Files persist until you restart the server or reboot

### Production (Google Cloud)
- Files stored in: `/tmp/` 
- Files persist for the duration of the instance lifetime
- ✓ Multiple requests can access the same file during a session
- ✓ Files automatically cleaned up when instance shuts down
- ⚠ Files are lost if a new instance handles the next request (due to scaling)

## Changes Made

1. **Simplified storage configuration**
   - Removed all GCS (Google Cloud Storage) code
   - Set `UPLOAD_FOLDER = '/tmp'` for all environments
   - Removed `google-cloud-storage` dependency

2. **Simplified helper functions**
   - `get_file_path(filename)` - Returns path in /tmp
   - `save_uploaded_file(file, filename)` - Saves to /tmp
   - `list_uploaded_files()` - Lists files in /tmp

3. **No setup required**
   - No GCS bucket needed
   - No additional configuration
   - Works immediately after deployment

## Usage

Files uploaded during a session will be available for analysis:

1. User uploads CSV file → Saved to `/tmp/filename_timestamp.csv`
2. User clicks analyze → File retrieved from `/tmp/`
3. Analysis completes → Results displayed
4. File remains in `/tmp/` for potential re-analysis
5. Instance shutdown → File automatically deleted

## Limitations

⚠ **Important**: Files are NOT persistent across:
- Instance restarts
- Scale-down and scale-up events
- Different instance handling requests (when min_instances < max_instances)

This is perfect for:
✓ Temporary analysis workflows
✓ Single-session use cases
✓ Development and testing

Not suitable for:
✗ Long-term file storage
✗ File history/archives
✗ Files needed across multiple sessions

## Deployment

No special configuration needed:

```powershell
gcloud app deploy
```

That's it! The app will work immediately with session-based file storage.

## Testing Locally

```powershell
python flask_server.py
```

Files will be stored in `/tmp/` and available for the entire development session.
