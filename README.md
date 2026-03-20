# Stable Diffusion Stylezer

Aplikasi web Flask untuk mengganti style/outfit pada gambar menggunakan Stable Diffusion Inpainting. User dapat:
1. Upload gambar yang ingin diubah
2. Mask area yang ingin diganti (draw pada canvas)
3. Upload gambar outfit/stylist (baju, hijab, celana, dll)
4. Klik Process → output gambar dengan style/outfit baru

## Setup Lokal

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

pip install -r requirements.txt
cp .env.example .env
# Edit .env untuk prompt dan config

python app.py
```

Buka http://localhost:5000

## Config (.env)

| Variable | Default | Deskripsi |
|----------|---------|-----------|
| SD_MODEL_ID | runwayml/stable-diffusion-inpainting | Model HuggingFace |
| SD_PROMPT | person wearing fashionable outfit... | Prompt default |
| SD_NEGATIVE_PROMPT | low quality, blurry... | Negative prompt |
| SD_NUM_INFERENCE_STEPS | 30 | Jumlah step inference |
| SD_GUIDANCE_SCALE | 7.5 | Guidance scale |
| SD_STRENGTH | 0.85 | Strength untuk inpainting |

## Colab / Kaggle

1. Push proyek ke GitHub (atau upload folder ke Colab/Kaggle)
2. Buka `run_colab.ipynb` (Google Colab) atau `run_kaggle.ipynb` (Kaggle)
3. Ubah URL git clone jika repo Anda berbeda
4. Daftar ngrok gratis di https://ngrok.com dan paste token
5. Jalankan semua cell
6. Buka URL ngrok yang muncul

**Colab:** Pilih Runtime > Change runtime type > GPU  
**Kaggle:** Settings > Accelerator > GPU

## Struktur

```
├── app.py              # Flask backend
├── templates/
│   └── index.html      # UI dengan masking canvas
├── requirements.txt
├── .env.example
├── run_colab.ipynb     # Notebook untuk Colab
├── run_kaggle.ipynb    # Notebook untuk Kaggle
└── README.md
```
