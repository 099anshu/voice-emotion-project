// Global variables
let mediaRecorder;
let audioChunks = [];
let audioBlob;
let recording = false;

// DOM Elements
const audioInput = document.getElementById('audioInput');
const uploadArea = document.getElementById('uploadArea');
const recordBtn = document.getElementById('recordBtn');
const recordingStatus = document.getElementById('recordingStatus');
const audioPlayerSection = document.getElementById('audioPlayerSection');
const audioPlayer = document.getElementById('audioPlayer');
const audioFilename = document.getElementById('audioFilename');
const replayBtn = document.getElementById('replayBtn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('resultsCard');
const historyList = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistory');

// File upload handling
audioInput.addEventListener('change', handleFileSelect);

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('audio/')) {
        handleAudioFile(files[0]);
    }
});

uploadArea.addEventListener('click', () => {
    audioInput.click();
});

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleAudioFile(file);
    }
}

async function handleAudioFile(file) {
    // Show audio player - use blob URL initially
    const audioUrl = URL.createObjectURL(file);
    audioPlayer.src = audioUrl;
    audioFilename.textContent = file.name;
    audioPlayerSection.style.display = 'block';
    resultsSection.style.display = 'none';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('audio', file);
    
    // Show loading
    loading.style.display = 'block';
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }
        
        const result = await response.json();
        
        console.log('📥 Received result from backend:', result);
        
        if (result.error) {
            alert('Error: ' + result.error);
            loading.style.display = 'none';
            return;
        }
        
        // Update audio player to use backend URL for consistent playback
        if (result.audio_url) {
            audioPlayer.src = result.audio_url;
        }
        
        // Display results
        displayResults(result);
        loading.style.display = 'none';
        resultsSection.style.display = 'block';
        
        // Refresh history after new recording
        loadHistory();
        
    } catch (error) {
        console.error('❌ Error:', error);
        alert('An error occurred: ' + error.message);
        loading.style.display = 'none';
    }
}

function displayResults(result) {
    console.log('🎨 Displaying results:', result);
    
    // Set transcription
    const transcriptionText = document.getElementById('transcriptionText');
    transcriptionText.textContent = result.transcription || 'No transcription available';
    
    // Set main emotion - ensure we use the exact emotion from backend
    const emotionName = document.getElementById('emotionName');
    const emotionEmoji = document.getElementById('emotionEmoji');
    const confidencePercentage = document.getElementById('confidencePercentage');
    const confidenceFill = document.getElementById('confidenceFill');
    
    // Get emotion directly from result (this is the exact emotion detected by backend)
    const detectedEmotion = result.emotion || 'unknown';
    const detectedEmoji = result.emoji || '🎭';
    
    console.log(`✅ Displaying emotion: ${detectedEmotion} with emoji: ${detectedEmoji}`);
    
    emotionName.textContent = detectedEmotion.charAt(0).toUpperCase() + detectedEmotion.slice(1);
    emotionEmoji.textContent = detectedEmoji;
    const confidence = (result.confidence * 100).toFixed(1);
    confidencePercentage.textContent = confidence + '%';
    confidenceFill.style.width = confidence + '%';
    
    // Display all emotions
    const emotionsGrid = document.getElementById('emotionsGrid');
    emotionsGrid.innerHTML = '';
    
    const emotionEmojis = {
        'neutral': '😐',
        'happy': '😊',
        'sad': '😢',
        'angry': '😠',
        'fear': '😨',
        'surprise': '😲'
    };
    
    // Sort emotions by score
    const sortedEmotions = Object.entries(result.all_scores)
        .sort((a, b) => b[1] - a[1]);
    
    sortedEmotions.forEach(([emotion, score]) => {
        const emotionItem = document.createElement('div');
        emotionItem.className = 'emotion-item';
        if (emotion === result.emotion) {
            emotionItem.classList.add('active');
        }
        
        const percentage = (score * 100).toFixed(1);
        emotionItem.innerHTML = `
            <div class="emotion-item-emoji">${emotionEmojis[emotion] || '🎭'}</div>
            <div class="emotion-item-name">${emotion}</div>
            <div class="emotion-item-score">${percentage}%</div>
        `;
        
        emotionsGrid.appendChild(emotionItem);
    });
    
    // Animate results
    resultsSection.style.animation = 'fadeInUp 0.6s ease-out';
}

// Recording functionality
recordBtn.addEventListener('click', toggleRecording);

async function toggleRecording() {
    if (!recording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            // Create audio blob
            const mimeType = mediaRecorder.mimeType || 'audio/webm';
            audioBlob = new Blob(audioChunks, { type: mimeType });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayer.src = audioUrl;
            audioFilename.textContent = 'Recording';
            audioPlayerSection.style.display = 'block';
            resultsSection.style.display = 'none';
            
            // Get file extension from mime type
            let extension = 'webm';
            if (mimeType.includes('webm')) extension = 'webm';
            else if (mimeType.includes('mp4')) extension = 'm4a';
            else if (mimeType.includes('ogg')) extension = 'ogg';
            
            // Create file with timestamp for unique filename
            const timestamp = Date.now();
            const filename = `recording_${timestamp}.${extension}`;
            const file = new File([audioBlob], filename, { type: mimeType });
            
            console.log(`🎙️ Recording stopped, sending file: ${filename}, type: ${mimeType}`);
            await handleAudioFile(file);
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        recording = true;
        recordBtn.classList.add('recording');
        recordBtn.querySelector('.record-text').textContent = 'Stop Recording';
        recordingStatus.textContent = '🔴 Recording... Speak now!';
        recordingStatus.style.color = '#DC143C';
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Could not access microphone. Please check permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && recording) {
        mediaRecorder.stop();
        recording = false;
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.record-text').textContent = 'Start Recording';
        recordingStatus.textContent = 'Recording stopped. Processing...';
        recordingStatus.style.color = '#8B4513';
    }
}

// Replay button
replayBtn.addEventListener('click', () => {
    audioPlayer.currentTime = 0;
    audioPlayer.play();
});

// History Management
const emotionEmojisMap = {
    'neutral': '😐',
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'surprise': '😲'
};

async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        const data = await response.json();
        
        if (data.error) {
            console.error('Error loading history:', data.error);
            return;
        }
        
        displayHistory(data.history || []);
    } catch (error) {
        console.error('Error loading history:', error);
    }
}

function displayHistory(history) {
    if (!historyList) return;
    
    if (history.length === 0) {
        historyList.innerHTML = '<p class="history-empty">No recordings yet. Start recording or upload a file!</p>';
        return;
    }
    
    historyList.innerHTML = '';
    
    history.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const date = new Date(item.timestamp);
        const formattedDate = date.toLocaleDateString('en-US', { 
            month: 'short', 
            day: 'numeric', 
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
        
        const confidencePercent = (item.confidence * 100).toFixed(1);
        const emoji = emotionEmojisMap[item.emotion] || '🎭';
        
        historyItem.innerHTML = `
            <div class="history-item-header">
                <div class="history-emotion">
                    <span class="history-emoji">${emoji}</span>
                    <span>${item.emotion.charAt(0).toUpperCase() + item.emotion.slice(1)}</span>
                </div>
                <div class="history-confidence">${confidencePercent}%</div>
            </div>
            <div class="history-filename">${item.filename}</div>
            ${item.transcription ? `<div class="history-transcription">"${item.transcription}"</div>` : ''}
            <div class="history-timestamp">${formattedDate}</div>
        `;
        
        // Make history item clickable to play audio
        historyItem.addEventListener('click', () => {
            if (item.audio_url) {
                audioPlayer.src = item.audio_url;
                audioPlayerSection.style.display = 'block';
                audioFilename.textContent = item.filename;
                audioPlayer.play();
            }
        });
        
        historyList.appendChild(historyItem);
    });
}

// Refresh history button
if (refreshHistoryBtn) {
    refreshHistoryBtn.addEventListener('click', () => {
        loadHistory();
    });
}

// Load history on page load
document.addEventListener('DOMContentLoaded', () => {
    loadHistory();
    
    // Add sparkle effect on hover for emotion items
    const emotionItems = document.querySelectorAll('.emotion-item');
    emotionItems.forEach(item => {
        item.addEventListener('mouseenter', () => {
            item.style.transition = 'all 0.3s ease';
        });
    });
});

