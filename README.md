# youtube-auto-shorts-generator

This is an AI-powered text-to-video generation tool that automatically creates videos from text topics. 

## AI Video Generator 🎬✨ 

#### An automated, AI-powered pipeline that creates engaging short videos from any text topic. Perfect for generating content for YouTube Shorts, TikTok, Reels, and social media. 

Takes a text topic as input (e.g., "dogs" or "space exploration") 
Generates a complete video with script, voiceover, background footage, and captions Outputs a finished MP4 video file 

### 🚀 Features 

  AI Script Writing: Generates concise, engaging scripts using OpenAI or OpenRouter.
  
  High-Quality Voiceover: Converts text to natural speech using Microsoft Edge TTS.
  
  Timed Captions: Automatically generates perfectly synchronized captions using AssemblyAI.
  
  Smart Visual Search: AI analyzes the script to find the most relevant video clips for each segment.
  
  Multi-Source Footage: Downloads high-quality videos from Pexels.

  Format Flexibility: Create both landscape videos and vertical shorts (--short flag).
  
  Fast Rendering: Utilizes MoviePy for efficient video composition and rendering.

## Prerequisites Python 3.8+ 
installed API Keys for the following services: 
AssemblyAI (Required for captions) Pexels (Required for video footage) OpenRouter or OpenAI (Required for script generation) 

## Installation 

    
Clone 
           
    git clone https://github.com/3d0n1/youtube-auto-shorts-generator.git
       
Run 

   cd youtube-auto-shorts-generator
       
#### Install Required Packages

    pip install openai edge-tts requests moviepy python-dotenv assemblyai 
       

#### Configure environment variables Edit a .env file in the root directory: 

  # Required: 
  
  Choose ONE AI provider 
    
         
    OPENROUTER_API_KEY=your_openrouter_key_here # OR 
         
    OPENAI_KEY=your_openai_key_here 
         
    # Required: For caption generation ASSEMBLYAI_KEY=your_assemblyai_key_here 
    # Required: For video footage PEXELS_KEY=your_pexels_key_here 
    # Optional: For text rendering on Windows IMAGEMAGICK_BINARY=C:/Path/To/ImageMagick/magick.exe 
        
## 🎯 Usage Basic Command 

    python auto_youtube.py "your topic here" 
         
#### Create Vertical Shorts 

    python auto_youtube.py "funny cats" --short 
        
#### Some Examples 

      # Educational content python auto_youtube.py "the science of nasa space explore" 
      # Fun facts python auto_youtube.py "interesting facts about dolphins" 
      # How-to content python auto_youtube.py "how to train your puppy" --short 
       
####  📁 Output Final Video: 

rendered_video.mp4 (in the root directory) 

Audio File: audio_tts.wav (can be deleted after processing)

Logs: API responses saved in .logs/ directory for debugging 

#### 🔧 Customization 

##### You can modify the following aspects by editing the constants in the script: 

Video Dimensions: Adjust default width/height for landscape or portrait 

AI Model: Switch between different OpenAI/OpenRouter models 

Voice Selection: Change the TTS voice (default: en-AU-WilliamNeural) 

Text Styling: Modify caption font, size, color, and position 

Video Quality: Adjust bitrate, codec, and rendering settings 

### ⭐ Star this repo if you found it useful!
