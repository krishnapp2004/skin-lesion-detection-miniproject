import os
from flask import Flask, render_template, request, redirect, session
from werkzeug.utils import secure_filename
from model_utils import predict_and_analyze
import json
import base64
from datetime import datetime

UPLOAD_FOLDER = "static/uploads"
HEATMAP_FOLDER = "static/heatmaps"

app = Flask(__name__)
app.secret_key = 'melanoma-detection-secret-key-2024'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------
# LOGIN
# ------------------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session['logged_in'] = True
        return redirect("/dashboard")
    return render_template("login.html")


@app.route("/logout")
def logout():
    """Logout route"""
    session.clear()
    return redirect("/")


# ------------------------
# PAGES
# ------------------------

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/education")
def education():
    return render_template("education.html")

@app.route("/scan")
def scan():
    return render_template("upload.html")


# ------------------------
# PREDICTION (WITH ERROR DETECTION)
# ------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle image upload and prediction
    WITH ERROR DETECTION for invalid images (dogs, cats, etc.)
    Saves heatmap to file instead of session
    """
    try:
        # Check if file is present
        file = request.files.get("image")
        if not file:
            print("❌ No file uploaded")
            return redirect("/scan")
        
        if file.filename == '':
            print("❌ Empty filename")
            return redirect("/scan")
        
        # Validate file type
        if not allowed_file(file.filename):
            print("❌ Invalid file type")
            return redirect("/scan")
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)
        print(f"✅ Image saved: {image_path}")
        
        # Run analysis WITH ERROR DETECTION
        print("🔬 Starting analysis...")
        results = predict_and_analyze(image_path)
        print("✅ Analysis complete!")
        
        # ===== ERROR DETECTION CHECK =====
        if results.get('error'):
            # Invalid image detected (dog, cat, random object, etc.)
            print(f"❌ Invalid image detected: {results.get('error_details')}")
            
            # Store minimal error info in session
            session['error_info'] = {
                'error_type': results.get('error_type'),
                'error_message': results.get('error_message'),
                'error_details': results.get('error_details'),
                'suggestion': results.get('suggestion'),
                'possible_reasons': results.get('possible_reasons'),
                'predicted_class': results.get('predicted_class'),
                'confidence': results.get('confidence')
            }
            
            # Clean up uploaded file
            try:
                os.remove(image_path)
                print("🗑️  Removed invalid image")
            except:
                pass
            
            # Render error page directly (NO grad-cam, NO pie chart)
            return render_template('error.html',
                error_type=results.get('error_type'),
                error_message=results.get('error_message'),
                error_details=results.get('error_details'),
                suggestion=results.get('suggestion'),
                possible_reasons=results.get('possible_reasons'),
                predicted_class=results.get('predicted_class'),
                confidence=results.get('confidence')
            )
        
        # ===== VALID IMAGE - PROCEED NORMALLY =====
        print("✅ Valid skin lesion image detected")
        
        # Extract and save heatmap to file (NOT session)
        heatmap_base64 = results['visualizations']['gradcam_heatmap']
        
        # Save heatmap as a separate file
        heatmap_filename = f"heatmap_{filename}"
        heatmap_path = os.path.join(HEATMAP_FOLDER, heatmap_filename)
        
        # Decode base64 and save
        try:
            # Extract base64 data (remove "data:image/png;base64," prefix)
            image_data = heatmap_base64.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            with open(heatmap_path, 'wb') as f:
                f.write(image_bytes)
            
            print(f"✅ Heatmap saved: {heatmap_path}")
            
            # Update results to use file path instead of base64
            results['visualizations']['gradcam_heatmap'] = f"/static/heatmaps/{heatmap_filename}"
        except Exception as e:
            print(f"⚠️  Could not save heatmap: {e}")
            # Fallback: use base64 (might cause session issues)
            pass
        
        # Store ONLY essential data in session (not the full results)
        session['analysis_results'] = results
        session['uploaded_image'] = filename
        
        # Render result page directly (avoid redirect to reduce session issues)
        return render_template("result.html", results=results)
    
    except Exception as e:
        import traceback
        print("\n❌ ERROR in prediction:")
        traceback.print_exc()
        return redirect("/scan")


@app.route("/results")
def results():
    """
    Alternative route to view results
    """
    results_data = session.get('analysis_results')
    
    if not results_data:
        return redirect("/scan")
    
    return render_template("result.html", results=results_data)


# ------------------------
# ERROR HANDLERS
# ------------------------

@app.errorhandler(413)
def file_too_large(e):
    """Handle file too large error"""
    return "File is too large. Maximum size is 16MB.", 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return "Page not found", 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return "Internal server error. Please try again.", 500


# ------------------------
# RUN
# ------------------------

if __name__ == "__main__":
    # Create directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(HEATMAP_FOLDER, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🏥 MELANOMA DETECTION SYSTEM WITH ERROR DETECTION")
    print("=" * 70)
    print("🌐 Server starting on http://localhost:5000")
    print("\n📋 Available Routes:")
    print("   /              → Login page")
    print("   /dashboard     → Dashboard")
    print("   /education     → Education page")
    print("   /scan          → Upload/Scan page")
    print("   /predict       → Process prediction (POST)")
    print("   /results       → View results")
    print("   /logout        → Logout")
    print("=" * 70)
    print("\n✅ Model loaded and ready!")
    print("📊 Features:")
    print("   • ✅ ERROR DETECTION (detects random objects)")
    print("   • ✅ CPU-optimized Grad-CAM heatmaps")
    print("   • ✅ Three-segment pie chart visualization")
    print("   • ✅ Smart risk assessment")
    print("   • ✅ All 9 lesion classes")
    print("   • ✅ Invalid image filtering (NO retraining needed!)")
    print("=" * 70)
    print("\n🔍 Error Detection Active:")
    print("   • Confidence threshold: 35%")
    print("   • Detects:cars, food, random objects....")
    print("   • Shows error page instead of results for invalid images")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)