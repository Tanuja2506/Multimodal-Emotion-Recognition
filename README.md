
# Multimodal Emotion Detection System

This project focuses on detecting human emotions using a multimodal approach, integrating data from textual, visual, and auditory inputs to achieve accurate and comprehensive emotion recognition.  

## Features

- Text Analysis:Uses Natural Language Processing (NLP) techniques to detect emotions in textual inputs.
- Visual Emotion Recognition: Utilizes computer vision to analyze facial expressions and gestures.
- Audio Analysis:Leverages audio processing techniques to recognize emotions from speech patterns and tone.
- Multimodal Fusion:Combines insights from text, audio, and visual data for enhanced accuracy in emotion detection.
- Customizable:Configurable to support various input formats and emotion models.


## Prerequisites
- Programming Languages:Python (preferred for NLP and ML pipelines)
Libraries & Frameworks:  
  - `TensorFlow` / `PyTorch` (for deep learning models)  
  - `OpenCV` (for image/video analysis)  
  - `librosa` (for audio analysis)  
  - `transformers` (for NLP tasks)  
  Environment: Python 3.8 or later


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/multimodal-emotion.git
   cd multimodal-emotion
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download pre-trained models for emotion detection (if applicable) from the `models` folder or external links provided.

## How It Works

1. Data Input:Accepts text, audio, and visual inputs. Inputs can be provided via live streams, uploaded files, or datasets.
2. Preprocessing: 
   - Text is tokenized and cleaned.  
   - Images are resized and normalized.  
   - Audio files are processed for feature extraction.  
3. Feature Extraction:Individual models extract features from each modality.
4. Fusion Layer: Combines extracted features to make a unified prediction.
5. Output:Returns the detected emotion(s) along with confidence scores.

## Usage

1. **Run the Application:**
   ```bash
   python main.py
   ```
2. **Provide Inputs:**  
   - Text: Input directly via the console or upload a `.txt` file.  
   - Visual: Upload images/videos or enable webcam mode.  
   - Audio: Upload `.wav`/`.mp3` files.  

3. **View Results:** Detected emotions will be displayed in the terminal or GUI (if enabled).

## Example

### Input:  
- Text:"I'm feeling so happy today!"  
- Image: [Image of a smiling person]  
- Audio:Speech sample with an excited tone.

### Output:  
```
Predicted Emotion: Joy
Confidence Scores: 
- Text: 0.90 
- Visual: 0.85 
- Audio: 0.88
```
## Applications

- Human-computer interaction
- Customer sentiment analysis
- Mental health monitoring
- Interactive voice/video systems

## Future Scope

- Integration with real-time systems (e.g., IoT devices)
- Support for additional languages and dialects
- Advanced fusion techniques for better accuracy

## Contributors

- Tanuja Shihare 
- Kalyani Waghchaure
- Sakshi Rampallewar  


