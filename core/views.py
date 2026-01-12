from django.shortcuts import render, redirect, get_object_or_404
from .forms import PatientForm
from .models import Patient
from .inference import detect_tumor, classify_tumor, segment_tumor, generate_gradcam_overlay
import hashlib
from django.db.models import Count
import json
import zipfile, os, tempfile
import plotly.graph_objs as go
import plotly.io as pio




# def compute_image_hash(image_file):
#     hasher = hashlib.sha256()
#     for chunk in image_file.chunks():
#         hasher.update(chunk)
#     return hasher.hexdigest()

# Home Page



def hash_image(file):
    file.seek(0)
    hash_val = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return hash_val

def home_view(request):
    return render(request, 'core/home.html')


# Patient Registration

def register_patient(request):
    import hashlib

    def hash_image(image_file):
        hasher = hashlib.sha256()
        for chunk in image_file.chunks():
            hasher.update(chunk)
        return hasher.hexdigest()

    error = None

    if request.method == 'POST':
        form = PatientForm(request.POST, request.FILES)
        if form.is_valid():
            name = form.cleaned_data['name']
            age = form.cleaned_data['age']
            gender = form.cleaned_data['gender']
            mri_image = request.FILES['mri_image']

            image_hash = hash_image(mri_image)

            duplicate = Patient.objects.filter(
                mri_hash=image_hash
            ).first()

            if duplicate:
                form = PatientForm()  
                error = "⚠️ This MRI scan has already been registered."
            else:
                patient = form.save(commit=False)
                patient.mri_hash = image_hash
                patient.save()
                return redirect('detect', pk=patient.pk)
    else:
        form = PatientForm()

    return render(request, 'core/register.html', {
        'form': form,
        'error': error
    })



# Combined Detection + Classification

def detection_view(request, pk):
    patient = get_object_or_404(Patient, pk=pk)

    try:
        image_path = patient.mri_image.path
        if not os.path.exists(image_path):
            return render(request, 'core/error.html', {
                'error': 'MRI image not found or not saved yet.'
            })
    except Exception as e:
        return render(request, 'core/error.html', {
            'error': f'Error accessing MRI image: {e}'
        })

    # Detection
    try:
        if not patient.detected:
            print("🧠 Running detection model...")
            detection_result = detect_tumor(image_path)
            print("🎯 Detection result:", detection_result)

            if detection_result in ['yes', 'no']:
                patient.detected = detection_result
                patient.save()
            else:
                raise ValueError("Invalid detection result returned from model.")
        else:
            print("📦 Using cached detection result:", patient.detected)
    except Exception as e:
        return render(request, 'core/error.html', {
            'error': f'Detection failed: {e}'
        })

    # Classification
    classification_result = None
    try:
        if patient.detected == "yes":
            if not patient.classified:
                print("🔬 Running classification model...")
                classification_result = classify_tumor(image_path)
                patient.classified = classification_result
                patient.save()
            else:
                classification_result = patient.classified
    except Exception as e:
        return render(request, 'core/error.html', {
            'error': f'Classification failed: {e}'
        })

    return render(request, 'core/detect.html', {
        'patient': patient,
        'classification': classification_result
    })


# Stage 3: Tumor Segmentation

def segment_view(request, pk):
    patient = get_object_or_404(Patient, pk=pk)

    try:
        image_path = patient.mri_image.path
        if not os.path.exists(image_path):
            return render(request, 'core/error.html', {
                'error': 'MRI image not found or not saved yet.'
            })
    except Exception as e:
        return render(request, 'core/error.html', {
            'error': f'Error accessing MRI image: {e}'
        })

    try:
        if not patient.segmented:
            seg_path, tumor_area, (center_x, center_y) = segment_tumor(image_path)
            if seg_path:
                patient.segmented = seg_path
                patient.tumor_area = tumor_area
                patient.tumor_center_x = center_x
                patient.tumor_center_y = center_y
                patient.save()
            else:
                raise ValueError("Segmentation failed to return a valid path.")
    except Exception as e:
        return render(request, 'core/error.html', {
            'error': f'Segmentation failed: {e}'
        })

    # GradCAM
    try:
        gradcam_path, confidence = generate_gradcam_overlay(image_path)
    except Exception as e:
        return render(request, 'core/error.html', {
            'error': f'GradCAM failed: {e}'
        })

    return render(request, 'core/segment.html', {
        'patient': patient,
        'seg_path': patient.segmented,
        'gradcam_path': gradcam_path,
        'confidence': f"{confidence:.3f}",
        'tumor_area': patient.tumor_area,
        'tumor_center': (patient.tumor_center_x, patient.tumor_center_y),
    })





def research_dashboard(request):
    patients = Patient.objects.all().order_by('-created_at')

    tumor_counts = list(Patient.objects.filter(detected='yes')
        .values('classified', 'gender')
        .annotate(count=Count('id')))

    detection_stats = list(Patient.objects.values('detected', 'gender')
        .annotate(count=Count('id')))

    return render(request, 'core/research.html', {
        'patients': patients,
        'tumor_counts': json.dumps(tumor_counts),
        'detection_stats': json.dumps(detection_stats),
    })

def batch_upload_view(request):
    if request.method == 'POST' and request.FILES.get('zipfile'):
        zip_file = request.FILES['zipfile']
        temp_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        image_paths = []
        for root, _, files in os.walk(temp_dir):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, filename))

        results = []
        for img_path in image_paths:
            try:
                detect = detect_tumor(img_path)
                classify = classify_tumor(img_path) if detect == 'yes' else None

                if detect == 'yes':
                    seg_path, area, center = segment_tumor(img_path)
                else:
                    seg_path, area, center = None, None, None

                results.append({
                    'filename': os.path.basename(img_path),
                    'detected': detect,
                    'classified': classify,
                    'area': round(area, 2) if area else None,
                    'center': str(center) if center else None,
                })

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

        request.session['batch_results'] = results
        return redirect('batch_dashboard')

    return render(request, 'core/batch_upload.html')



def batch_dashboard_view(request):
    results = request.session.get('batch_results', [])

    # Bar chart data
    classified_counts = {}
    for r in results:
        if r['classified']:
            classified_counts[r['classified']] = classified_counts.get(r['classified'], 0) + 1
    bar_fig = go.Figure([go.Bar(x=list(classified_counts.keys()), y=list(classified_counts.values()))])
    bar_html = pio.to_html(bar_fig, full_html=False, include_plotlyjs='cdn')

    # Pie chart data
    detected_counts = {'yes': 0, 'no': 0}
    for r in results:
        detected_counts[r['detected']] += 1
    pie_fig = go.Figure([go.Pie(labels=list(detected_counts.keys()), values=list(detected_counts.values()))])
    pie_html = pio.to_html(pie_fig, full_html=False, include_plotlyjs=False)

    return render(request, 'core/batch_dashboard.html', {
        'results': results,
        'bar_chart': bar_html,
        'pie_chart': pie_html
    })



