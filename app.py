import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import time
import io

print("Loading model...")

# ==========================================================
# LOAD MODEL
# ==========================================================
MODEL_PATH = "best_model.pth"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

model = models.mobilenet_v3_large(weights=None)

# Classifier untuk 5 kelas
in_features = model.classifier[3].in_features
model.classifier = nn.Sequential(
    model.classifier[0],
    model.classifier[1],
    nn.Dropout(0.2),
    nn.Linear(in_features, 5)
)

state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully")

# ==========================================================
# TRANSFORM
# ==========================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# CLASS & TREATMENT
# ==========================================================
class_names = ["Bercak Daun", "Bulai", "Daun Sehat", "Hawar Daun", "Karat Daun"]

treatment_info = {
    "Bercak Daun": (
        "• Aplikasi kompos *Trichoderma reesei* sebagai agen hayati.\n"
        "• Menghambat perkembangan patogen melalui kompetisi, antibiosis, dan induksi ketahanan sistemik.\n"
        "• Membantu memperlambat intensitas bercak daun dan menekan penyebaran penyakit."
    ),

    "Bulai": (
        "a. Penanaman Serempak\n"
        "   • Mengurangi keberlangsungan patogen dengan memutus ketersediaan inang.\n\n"
        "b. Pembersihan Lahan (Sanitasi)\n"
        "   • Membakar atau mengubur sisa tanaman terinfeksi agar inokulum tidak bertahan lama.\n\n"
        "c. Pemusnahan Tanaman Sakit (Roguing)\n"
        "   • Mencabut dan memusnahkan tanaman bergejala sejak dini.\n\n"
        "d. Penggunaan Varietas Tahan\n"
        "   • Varietas yang direkomendasikan: R7, P31, P35.\n"
        "   • Gunakan benih bersertifikat.\n\n"
        "e. Perlakuan Benih (Seed Treatment)\n"
        "   • Melindungi benih dari infeksi sistemik.\n"
        "   • Tiga metode:\n"
        "       1. Tanpa fungisida: rendam 30 menit → jemur.\n"
        "       2. Dengan fungisida: taburkan dimetomorf/fenamidone.\n"
        "       3. Benih pabrikan: sudah dilapisi fungisida (umumnya metalaksil).\n"
        "   • Rotasi fungisida penting untuk mencegah resistensi.\n\n"
        "f. Tidak Menyemprot Tanaman Terinfeksi\n"
        "   • Fungisida tidak efektif untuk mengobati bulai pada tanaman yang sudah sakit.\n"
        "   • Fokus utama: seed treatment & roguing."
    ),

    "Daun Sehat": (
        "• Tanaman dalam kondisi sehat.\n"
        "• Lanjutkan penyiraman teratur.\n"
        "• Berikan pemupukan sesuai kebutuhan.\n"
        "• Lakukan pengendalian gulma secara rutin."
    ),

    "Hawar Daun": (
        "• Gunakan varietas jagung yang tahan *Exserohilum turcicum*.\n"
        "• Kendalikan organisme penyebar patogen.\n"
        "• Lakukan pemupukan dan aplikasi pestisida sesuai dosis.\n"
        "• Aplikasi kompos *Trichoderma reesei* untuk menekan intensitas penyakit."
    ),

    "Karat Daun": (
        "• Pastikan nutrisi tanaman seimbang.\n"
        "• Kendalikan kondisi lembap yang berlebih.\n"
        "• Lakukan pemantauan rutin gejala awal.\n"
        "• Gunakan agen hayati seperti *Trichoderma reesei* untuk meningkatkan ketahanan tanaman."
    ),

    "Unknown": (
        "Gambar tidak jelas.\n"
        "Pastikan foto terang, fokus, dan objek daun terlihat jelas."
    )
}

CONFIDENCE_THRESHOLD = 0.55

# ==========================================================
# FLASK APP
# ==========================================================
app = Flask(__name__)


def allowed_file(filename):
    """Validasi ekstensi file."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "usage": "POST ke /predict dengan key 'file'"
    })


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    if "file" not in request.files:
        return jsonify({"error": "Gunakan key 'file' untuk upload gambar."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Format harus .jpg/.jpeg/.png"}), 400

    try:
        # ---------------------------- Load Image ----------------------------
        file_bytes = file.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        # ---------------------------- Prediction ----------------------------
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        # ---------------------------- Get Top-1 & Top-3 ----------------------------
        max_conf = float(probs.max())
        pred_idx = int(probs.argmax())

        # Urutkan dari terbesar ke terkecil
        sorted_probs = sorted(probs, reverse=True)
        top1 = sorted_probs[0]
        top2 = sorted_probs[1]

        # ---------------------------- Unknown Conditions ----------------------------
        # Kondisi 1: Confidence sangat rendah
        THRESHOLD = 0.85

        # Kondisi 2: Model bingung (selisih tipis antara top-1 & top-2)
        MARGIN = 0.15     # semakin kecil → semakin sensitif

        if (top1 < THRESHOLD) or ((top1 - top2) < MARGIN):
            predicted_class = "Unknown"
        else:
            predicted_class = class_names[pred_idx]

        # ---------------------------- Top-3 result ----------------------------
        top3_idx = probs.argsort()[-3:][::-1]
        top3 = [{"class": class_names[i]} for i in top3_idx]

        response_time = round(time.time() - start_time, 4)

        return jsonify({
            "id": f"rec_{uuid.uuid4().hex[:12]}",
            "predicted_class": predicted_class,
            "confidence": round(max_conf, 4),
            "threshold": THRESHOLD,
            "margin": MARGIN,
            "is_unknown": predicted_class == "Unknown",
            "treatment": treatment_info.get(predicted_class, ""),
            "top3_predictions": top3,
            "response_time_sec": response_time
        })

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500



if __name__ == "__main__":
    print("Server running at http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)

