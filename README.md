# Image Caption Generator using CNN + LSTM

## 📌 Project Overview

This project implements an **Image Caption Generator** using **Deep Learning** techniques. The model automatically generates meaningful captions for images by combining **Computer Vision** and **Natural Language Processing (NLP)**.

The system uses a **Convolutional Neural Network (CNN)** to extract visual features from images and a **Long Short-Term Memory (LSTM)** network to generate natural language descriptions.

---

## 🚀 Features

* Automatically generates captions for input images
* Uses **CNN for image feature extraction**
* Uses **LSTM for sequential text generation**
* Trained on an image-caption dataset
* Evaluated using **BLEU Score**

---

## 🧠 Model Architecture

1. **CNN (Feature Extractor)**

   * Pretrained CNN model extracts important visual features from images.

2. **LSTM (Caption Generator)**

   * Processes extracted image features and generates captions word by word.

3. **Tokenizer**

   * Converts words into numerical sequences for model training.

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

---

## 📂 Project Structure

```
Image-Caption-Generator/
│
├── dataset/
│   ├── images/
│   └── captions.txt
│
├── model/
│   ├── model.h5
│
├── notebooks/
│   └── image_caption_generator.ipynb
│
├── app.py (optional for Streamlit deployment)
│
└── README.md
```



## ▶️ How to Run

1. Open the Jupyter Notebook:

```bash
jupyter notebook
```

2. Run the notebook:

```
image_caption_generator.ipynb
```

3. Upload an image to generate a caption.

---

## 📊 Model Evaluation

The model performance is evaluated using **BLEU Score**, which measures the similarity between generated captions and reference captions.

Example:

```
BLEU Score: 0.52
```

---

## 📷 Example Output

Input Image → A man riding a surfboard on a wave
Generated Caption → **"A surfer riding a wave on the ocean."**

---

## 🔮 Future Improvements

* Improve caption quality with **Transformer models**
* Train on larger datasets
* Deploy the model as a **web application**

---

## 👩‍💻 Author

**Kajal Yerone**

If you like this project, feel free to ⭐ the repository!
