import uvicorn
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    print("\n" + "="*50)
    print("AI RESUME SCREENING DASHBOARD")
    print("="*50)
    print("1. Server is launching on: http://localhost:8000")
    print("2. Keep this terminal OPEN while using the app.")
    print("3. If you see 'Site can't be reached', try http://127.0.0.1:8000")
    print("="*50 + "\n")
    
    try:
        uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
    except Exception as e:
        print(f"\n[ERROR] Failed to start server: {e}")
        print("Check if Port 8000 is already in use by another application.")
