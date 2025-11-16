Understand the RAVDESS filename codes (Critical) : 03-01-05-01-02-01-12.wav

The third number tells the emotion:
3rd Number	Emotion
01	neutral
02	calm (optional, ignore)
03	happy
04	sad
05	angry
06	fear
07	disgust (we skip)
08	surprise

#steps to clone

-git clone https://github.com/<your-username>/voice-emotion-project.git
-cd voice-emotion-project
-python3 -m venv venv
-source venv/bin/activate   # Mac
-pip install -r requirements.txt
-python app.py
