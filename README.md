# Understanding RAVDESS Filename Codes (Critical)

Example filename: `03-01-05-01-02-01-12.wav`

### 📌 Emotion Code (3rd number)
The **third number** in the filename represents the emotion:

| Code | Emotion  |
|------|----------|
| 01   | Neutral  |
| 02   | Calm *(optional, can ignore)* |
| 03   | Happy    |
| 04   | Sad      |
| 05   | Angry    |
| 06   | Fear     |
| 07   | Disgust *(skipped)* |
| 08   | Surprise |

---

# 🚀 Steps to Clone & Run the Project

```bash
# 1. Clone the repository
git clone https://github.com/voice-emotion-project.git

# 2. Go to project folder
cd voice-emotion-project

# 3. Create a virtual environment
python3 -m venv venv

# 4. Activate the virtual environment (Mac)
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Run the app
python app.py

