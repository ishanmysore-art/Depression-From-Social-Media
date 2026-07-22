# 🧠 Adolescent Depression Detection via Social Media Linguistic Markers

## 📖 Overview
This project introduces a machine learning-driven predictive model designed to analyze social media activity and detect early indicators of depression in adolescents. By analyzing a user's recent posts on platforms like Twitter and Instagram, the tool evaluates linguistic markers to determine an overall depression risk level. It aims to bridge AI technology with digital support systems by suggesting relevant mental health resources and encouraging professional diagnosis when appropriate. 

## ✨ Features
- **📱 Multi-Platform Scraping**: Automatically scrapes a user's 100 most recent posts from Twitter (X) or Instagram using their public handle.
- **📝 Advanced Text Preprocessing**: Cleans data by removing URLs, HTML tags, and punctuation, while normalizing chat abbreviations and converting emojis to textual representations. Optical Character Recognition (OCR) is used to extract text from Instagram images.
- **🧠 Bayesian-Optimized MLP Model**: Utilizes a highly optimized Multilayer Perceptron (MLP) Neural Network (achieving a 92.2% F2 Score) to assign a depression risk probability to each post.
- **📊 Risk Level Categorization**: Aggregates post scores to classify users into four risk categories: Severe, Moderate, Mild, or At risk for no depression.
- **🤝 Actionable Resources**: Provides immediate, tailored recommendations and links to reputable mental health resources (e.g., WHO, American Psychiatric Association, Scout) based on the predicted risk level.

## 🛠️ Technologies Used
- **Backend & Machine Learning**: Python, Scikit-Learn (TF-IDF Vectorizer), TensorFlow/Keras (MLP Neural Network), XGBoost, cuML (Random Forest)
- **Hyperparameter Tuning**: Bayesian Optimization (`bayes_opt`)
- **Data Scraping & Extraction**: `twikit` (Twitter), `instaloader` (Instagram), `pytesseract` (OCR)
- **NLP Preprocessing**: `nltk`, `BeautifulSoup`, `emoji`
- **Web Deployment**: Flask, Gunicorn, HTML/CSS

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/depression-marker-detection.git
   cd depression-marker-detection
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have Tesseract OCR installed on your system for `pytesseract` to function properly).*

## 💻 Usage

1. **Start the Flask server:**
   ```bash
   python app.py
   ```
   *(Alternatively, you can use Gunicorn for production deployments).*

2. **Access the Web Interface:**
   Open your web browser and navigate to `http://localhost:5000`.

3. **Run a Prediction:**
   - Select the platform (Twitter or Instagram).
   - Enter a **public** account handle.
   - Click submit. The application will scrape the posts, analyze the linguistic markers, and output a probabilistic risk assessment along with suggested resources.
   
*Disclaimer: This application should never be used as a substitute for professional medical evaluation or treatment. The predictions are probabilistic and designed to highlight the need for a thorough evaluation by a healthcare provider.*

## 🤝 Contributing
Contributions to improve the model, expand platform support, or enhance the web interface are welcome! 
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
