# 👗 Fashion Recommendation System  
A deep-learning powered **content-based image recommendation system** built using **TensorFlow (ResNet50)** for feature extraction and **k-Nearest Neighbors (kNN)** for similarity search.  
The web interface is built using **Streamlit**, allowing users to upload an image and get visually similar fashion items instantly.
Kaggle Dataset link - https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images

---

## 🚀 Features
- 🧠 **Deep Learning Feature Extraction** using ResNet50 pretrained on ImageNet  
- ⚡ **Fast Similarity Search** using kNN (Euclidean distance)  
- 📸 **Upload any fashion image** to get 5 similar recommendations  
- 🌐 **Interactive Web App** with a clean, responsive UI  
- 💾 **Feature Embeddings Stored** for fast inference  
- 🔥 Ready for deployment on platforms like Streamlit Cloud, Render, etc.

---

## 📂 Project Structure
```
├── app.py # Streamlit main application
├── Images_features.pkl # Extracted image embeddings (NumPy array)
├── filenames.pkl # List of image file paths
├── upload/ # Temporary uploaded images
├── images/ # Dataset images
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## 🛠️ Tech Stack
### **Machine Learning & DL**
- TensorFlow / Keras  
- ResNet50  
- NumPy  
- Scikit-Learn (NearestNeighbors)

### **Frontend / Interface**
- Streamlit

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

```
## 🧠 How It Works

🔹 Step 1 — Feature Extraction

ResNet50 (without top layers) extracts a 2048-dimension feature vector for each fashion image.

🔹 Step 2 — Feature Normalization

L2-normalization is applied to make distance comparison effective.

🔹 Step 3 — Similarity Search

Using kNN (n_neighbors=6, metric=euclidean), we find the nearest images.

🔹 Step 4 — Display Results

The app shows the top 5 most similar items visually.

🖼️ App Preview

<img width="1901" height="873" alt="image" src="https://github.com/user-attachments/assets/fbdce44a-c6f9-4418-817d-b7d78be0866c" />

<img width="1905" height="872" alt="image" src="https://github.com/user-attachments/assets/57226d23-a0ca-405c-97ad-dd6dc78cdc5b" />


<img width="1908" height="864" alt="image" src="https://github.com/user-attachments/assets/9cc449db-b71b-48d3-acc8-75c580e95535" />
