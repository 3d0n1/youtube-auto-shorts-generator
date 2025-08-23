#!/usr/bin/env python3
# Consolidated Video Generator - All functionality in one file

import os
from dotenv import load_dotenv
load_dotenv()

imagemagick_path = os.getenv("IMAGEMAGICK_BINARY")
if imagemagick_path:
    os.environ["IMAGEMAGICK_BINARY"] = imagemagick_path
else:
    print("Warning: IMAGEMAGICK_BINARY not found in .env file. Text rendering will be disabled.")


# Core Imports
from openai import OpenAI
import edge_tts
import json
import asyncio
import argparse
import time
import tempfile
import zipfile
import platform
import subprocess
import requests
import re
import assemblyai as aai
from datetime import datetime
from typing import List, Tuple
from moviepy.editor import (AudioFileClip, CompositeVideoClip, CompositeAudioClip, 
                           ImageClip, TextClip, VideoFileClip)
from moviepy.audio.fx.audio_loop import audio_loop
from moviepy.audio.fx.audio_normalize import audio_normalize



# ========================
# Configuration & Constants
# ========================
VIDEO_SERVER = "pexel"
LOG_TYPE_GPT = "GPT"
LOG_TYPE_PEXEL = "PEXEL"
DIRECTORY_LOG_GPT = ".logs/gpt_logs"
DIRECTORY_LOG_PEXEL = ".logs/pexel_logs"

# Configure AssemblyAI
aai.settings.api_key = os.getenv("ASSEMBLYAI_KEY")

# ========================
# Utility Functions
# ========================
def log_response(log_type, query, response):
    """Log API responses to files"""
    log_entry = {
        "query": query,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }
    
    if log_type == LOG_TYPE_GPT:
        os.makedirs(DIRECTORY_LOG_GPT, exist_ok=True)
        filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_gpt3.txt'
        filepath = os.path.join(DIRECTORY_LOG_GPT, filename)
        with open(filepath, "w") as outfile:
            outfile.write(json.dumps(log_entry) + '\n')
    
    elif log_type == LOG_TYPE_PEXEL:
        os.makedirs(DIRECTORY_LOG_PEXEL, exist_ok=True)
        filename = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_pexel.txt'
        filepath = os.path.join(DIRECTORY_LOG_PEXEL, filename)
        with open(filepath, "w") as outfile:
            outfile.write(json.dumps(log_entry) + '\n')

def download_file(url, filename):
    """Download a file from URL"""
    with open(filename, 'wb') as f:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.1;) AppleWebKit/601.49 (KHTML, like Gecko) Chrome/53.0.2428.140 Safari/601"
        }
        response = requests.get(url, headers=headers)
        f.write(response.content)

def search_program(program_name):
    """Search for a program in system PATH"""
    try: 
        search_cmd = "where" if platform.system() == "Windows" else "which"
        return subprocess.check_output([search_cmd, program_name]).decode().strip()
    except subprocess.CalledProcessError:
        return None

def get_program_path(program_name):
    """Get full path to a program"""
    return search_program(program_name)

def fix_json(json_str):
    """Fix common JSON formatting issues"""
    return (json_str
            .replace("'", '"')
            .replace("'", "'")
            .replace(""", '"').replace(""", '"')
            .replace("'", "'").replace("'", "'"))

# ========================
# Script Generation
# ========================
SYSTEM_PROMPT = """You are a YouTube Shorts content writer. Create a concise script (under 50 seconds/~140 words).

STRICT REQUIREMENTS:
1. Output MUST be valid JSON only
2. Use exactly this format:
{"script": "Your script text here"}

EXAMPLE OUTPUT:
{"script": "Did you know dogs can smell 10,000-100,000 times better than humans? Their wet noses help capture scent molecules. The Bloodhound's sense of smell is so accurate it's admissible in court!"}"""

try:
    # Initialize OpenAI/OpenRouter client
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_KEY")
    
    if openrouter_key:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",
                "X-Title": "Text-To-Video-AI"
            }
        )
        model = "anthropic/claude-3-haiku"
        print("Configured for OpenRouter with model:", model)
    elif openai_key:
        model = "gpt-4-turbo"
        client = OpenAI(api_key=openai_key)
        print("Configured for OpenAI with model:", model)
    else:
        raise ValueError("No API keys found in .env file")
        
    # Initialize AssemblyAI
    aai.settings.api_key = os.getenv("ASSEMBLYAI_KEY")
    if not aai.settings.api_key:
        raise ValueError("ASSEMBLYAI_KEY not found in .env file")

except Exception as e:
    print(f"\nERROR: Failed to initialize API clients: {e}")
    print("Please check your .env file contains:")
    print("OPENROUTER_API_KEY=your_key_here")
    print("or OPENAI_KEY=your_key_here")
    print("and ASSEMBLYAI_KEY=your_key_here")
    exit(1)

def generate_script(topic, max_retries=3):
    """Generate video script using AI"""
    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1}: Generating script for '{topic}'")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Topic: {topic}\n\nReturn JSON only, no commentary"}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                timeout=30
            )
            
            if not response.choices:
                print("API returned empty response")
                continue
                
            content = response.choices[0].message.content
            print("Raw response preview:", content[:150] + "...")
            
            try:
                result = json.loads(content)
                if "script" not in result:
                    raise ValueError("Missing 'script' key in response")
                return result["script"]
            except json.JSONDecodeError:
                json_str = content[content.find('{'):content.rfind('}')+1]
                result = json.loads(json_str)
                return result["script"]
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed:", str(e))
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise RuntimeError(f"Failed after {max_retries} attempts") from e
    raise RuntimeError("Max retries reached without success")

# ========================
# Audio Generation
# ========================
async def generate_audio(text, outputFilename):
    """Generate audio from text using edge-tts"""
    communicate = edge_tts.Communicate(text, "en-AU-WilliamNeural")
    await communicate.save(outputFilename)

# ========================
# Caption Generation
# ========================
def process_transcript_to_captions(transcript, maxCaptionSize: int = 15) -> List[Tuple[Tuple[float, float], str]]:
    """Process transcript into timed captions"""
    captions = []
    current_caption = []
    current_start = 0
    
    for word in transcript.words:
        word_text = word.text
        word_start = word.start / 1000  # Convert ms to seconds
        
        if not current_caption:
            current_start = word_start
        
        test_caption = ' '.join(current_caption + [word_text])
        if len(test_caption) <= maxCaptionSize:
            current_caption.append(word_text)
        else:
            if current_caption:
                captions.append((
                    (current_start, word_start),
                    ' '.join(current_caption)
                ))
            current_caption = [word_text]
            current_start = word_start
    
    if current_caption:
        captions.append((
            (current_start, transcript.words[-1].end / 1000),
            ' '.join(current_caption)
        ))
    
    return captions

def generate_timed_captions(audio_filename: str, model_size: str = None) -> List[Tuple[Tuple[float, float], str]]:
    """Generate timed captions using AssemblyAI API"""
    try:
        transcriber = aai.Transcriber()
        config = aai.TranscriptionConfig(
            punctuate=True,
            format_text=True,
            speaker_labels=False
        )
        
        transcript = transcriber.transcribe(audio_filename, config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"Transcription failed: {transcript.error}")
        if not transcript.words:
            raise Exception("No words were transcribed")
        
        return process_transcript_to_captions(transcript)
    
    except Exception as e:
        print(f"Error in AssemblyAI caption generation: {e}")
        return [((0, 1), "Caption generation failed")]

# ========================
# Video Search Query Generation
# ========================
VIDEO_SEARCH_PROMPT = """# STRICT JSON OUTPUT REQUIRED
Generate video search keywords in this exact format:
{
  "segments": [
    {
      "time_range": [t1, t2],
      "keywords": ["keyword1", "keyword2", "keyword3"]
    }
  ]
}

Guidelines:
1. Create 3 VISUAL keywords per segment (2-4 seconds each)
2. Must be concrete (e.g., "dog running" not "happy moment")
3. English only, 1-2 words per keyword
4. Cover entire video duration
5. No additional text or explanations

Example Input: "The cheetah runs at 75 mph"
Example Output: {
  "segments": [
    {
      "time_range": [0, 2],
      "keywords": ["cheetah running", "fast animal", "75 mph"]
    }
  ]
}"""

def call_OpenAI(script, captions_timed):
    """Call OpenAI API for video search queries"""
    user_content = f"""Script: {script}
Timed Captions: {json.dumps(captions_timed)}"""
    
    print("Sending to AI:", user_content[:200] + "...")
    
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VIDEO_SEARCH_PROMPT},
            {"role": "user", "content": user_content}
        ],
        timeout=30
    )
    
    text = response.choices[0].message.content.strip()
    text = re.sub(r'\s+', ' ', text)
    log_response(LOG_TYPE_GPT, script, text)
    return text

def getVideoSearchQueriesTimed(script, captions_timed, max_retries=3):
    """Get timed video search queries from AI"""
    end = captions_timed[-1][0][1]
    out = [[[0, 0], []]]
    
    for attempt in range(max_retries):
        try:
            while out[-1][0][1] < end:
                content = call_OpenAI(script, captions_timed)
                
                try:
                    data = json.loads(content)
                    if "segments" in data:
                        new_segments = [
                            [[seg["time_range"][0], seg["time_range"][1]], seg["keywords"]]
                            for seg in data["segments"]
                        ]
                        out.extend(new_segments)
                        continue
                except json.JSONDecodeError:
                    pass
                
                try:
                    fixed_content = fix_json(content.replace("```json", "").replace("```", ""))
                    segments = json.loads(fixed_content)
                    out.extend(segments)
                except Exception as e:
                    print(f"Attempt {attempt + 1} parse error:", e)
                    print("Raw content:", content[:200] + "...")
                    raise
                    
            return out[1:]  # Skip initial dummy segment
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed:", str(e))
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed after {max_retries} attempts") from e
            continue
    return None

def merge_empty_intervals(segments):
    """Merge consecutive empty intervals in video segments"""
    merged = []
    i = 0
    while i < len(segments):
        interval, url = segments[i]
        if url is None:
            j = i + 1
            while j < len(segments) and segments[j][1] is None:
                j += 1
            if i > 0:
                prev_interval, prev_url = merged[-1]
                if prev_url is not None and prev_interval[1] == interval[0]:
                    merged[-1] = [[prev_interval[0], segments[j-1][0][1]], prev_url]
                else:
                    merged.append([interval, prev_url])
            else:
                merged.append([interval, None])
            i = j
        else:
            merged.append([interval, url])
            i += 1
    return merged

# ========================
# Background Video Generation
# ========================
def search_videos(query_string, orientation_landscape=True):
    """Search for videos on Pexels"""
    PEXELS_API_KEY = os.getenv('PEXELS_KEY')
    url = "https://api.pexels.com/videos/search"
    headers = {
        "Authorization": PEXELS_API_KEY,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {
        "query": query_string,
        "orientation": "landscape" if orientation_landscape else "portrait",
        "per_page": 15
    }

    response = requests.get(url, headers=headers, params=params)
    json_data = response.json()
    log_response(LOG_TYPE_PEXEL, query_string, response.json())
    return json_data

def getBestVideo(query_string, orientation_landscape=True, used_vids=[]):
    """Get the best matching video from Pexels"""
    vids = search_videos(query_string, orientation_landscape)
    videos = vids['videos']

    if orientation_landscape:
        filtered_videos = [video for video in videos if video['width'] >= 1920 and video['height'] >= 1080]
    else:
        filtered_videos = [video for video in videos if video['width'] >= 1080 and video['height'] >= 1920]

    sorted_videos = sorted(filtered_videos, key=lambda x: abs(15-int(x['duration'])))

    for video in sorted_videos:
        for video_file in video['video_files']:
            if orientation_landscape:
                if video_file['width'] >= 1920 and video_file['height'] >= 1080:
                    if not (video_file['link'].split('.hd')[0] in used_vids):
                        return video_file['link']
            else:
                if video_file['width'] >= 1080 and video_file['height'] >= 1920:
                    if not (video_file['link'].split('.hd')[0] in used_vids):
                        return video_file['link']
    print("NO LINKS found for this round of search with query:", query_string)
    return None

def generate_video_url(timed_video_searches, video_server, is_short=False):
    """Generate video URLs for all timed segments"""
    timed_video_urls = []
    if video_server == "pexel":
        used_links = []
        for (t1, t2), search_terms in timed_video_searches:
            url = ""
            for query in search_terms:
                url = getBestVideo(query, orientation_landscape=not is_short, used_vids=used_links)
                if url:
                    used_links.append(url.split('.hd')[0])
                    break
            timed_video_urls.append([[t1, t2], url])
    elif video_server == "stable_diffusion":
        timed_video_urls = get_images_for_video(timed_video_searches)
    return timed_video_urls

# ========================
# Video Rendering
# ========================
def get_output_media(audio_file_path, timed_captions, background_video_data, video_server):
    """Render final video with all components"""
    OUTPUT_FILE_NAME = "rendered_video.mp4"
    magick_path = get_program_path("magick")
    print(magick_path)
    if magick_path:
        os.environ['IMAGEMAGICK_BINARY'] = magick_path
    else:
        os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'
    
    visual_clips = []
    for (t1, t2), video_url in background_video_data:
        video_filename = tempfile.NamedTemporaryFile(delete=False).name
        download_file(video_url, video_filename)
        
        video_clip = VideoFileClip(video_filename)
        video_clip = video_clip.set_start(t1)
        video_clip = video_clip.set_end(t2)
        visual_clips.append(video_clip)
    
    audio_clips = []
    audio_file_clip = AudioFileClip(audio_file_path)
    audio_clips.append(audio_file_clip)

    for (t1, t2), text in timed_captions:
        text_clip = TextClip(txt=text, fontsize=100, color="white", 
                            stroke_width=3, stroke_color="black", method="label")
        text_clip = text_clip.set_start(t1)
        text_clip = text_clip.set_end(t2)
        text_clip = text_clip.set_position(["center", 800])
        visual_clips.append(text_clip)

    video = CompositeVideoClip(visual_clips)
    
    if audio_clips:
        audio = CompositeAudioClip(audio_clips)
        video.duration = audio.duration
        video.audio = audio

    video.write_videofile(OUTPUT_FILE_NAME, codec='libx264', 
                         audio_codec='aac', fps=25, preset='veryfast')
    
    for (t1, t2), video_url in background_video_data:
        video_filename = tempfile.NamedTemporaryFile(delete=False).name
        os.remove(video_filename)

    return OUTPUT_FILE_NAME

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a topic.")
    parser.add_argument("topic", type=str, help="The topic for the video")
    parser.add_argument("--short", action="store_true", help="Generate vertical videos for shorts")
    args = parser.parse_args()
    
    SAMPLE_TOPIC = args.topic
    SAMPLE_FILE_NAME = "audio_tts.wav"
    VIDEO_SERVER = "pexel"

    response = generate_script(SAMPLE_TOPIC)
    print("script: {}".format(response))

    asyncio.run(generate_audio(response, SAMPLE_FILE_NAME))

    timed_captions = generate_timed_captions(SAMPLE_FILE_NAME)
    print(timed_captions)

    search_terms = getVideoSearchQueriesTimed(response, timed_captions)
    print(search_terms)

    background_video_urls = None
    if search_terms is not None:
        background_video_urls = generate_video_url(search_terms, VIDEO_SERVER, is_short=args.short)
        print(background_video_urls)
    else:
        print("No background video")
        
    background_video_urls = merge_empty_intervals(background_video_urls)

    if background_video_urls is not None:
        video = get_output_media(SAMPLE_FILE_NAME, timed_captions, background_video_urls, VIDEO_SERVER)
        print(video)
    else:
        print("No video")