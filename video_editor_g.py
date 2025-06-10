import os
import json
import tempfile
import uuid
import time
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
import cv2
import numpy as np

# Load environment variables from .env file
print("[startup] Loading environment variables")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"[startup] Retrieved OPENAI_API_KEY: {'set' if OPENAI_API_KEY else 'not set'}")

# Initialize FastAPI app
print("[startup] Initializing FastAPI app")
app = FastAPI(title="Professional Video Editor API")

# Import OpenAI
print("[startup] Attempting to initialize OpenAI client")
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("[startup] OpenAI client initialized successfully")
except ImportError:
    print("Warning: OpenAI not available. Install with: pip install openai")
    client = None
    print("[startup] OpenAI client set to None due to ImportError")

# Define video editing parameters

class VideoEditParams(BaseModel):
    operation: str
    cut_duration: float
    replace_duration: Optional[float] = None
    cut_location: Optional[str] = "start"
    replace_location: Optional[str] = "start"

def find_best_scene_cut(
    clip,
    target_time: float,
    fps: int,
    window_seconds: float = 3.0,
    threshold_hist: float = 0.4,
    threshold_pix: float = 0.1,
):
    """
    Improved scene-cut detection using separate histogram and pixel difference thresholds.
    """
    print(f"[find_best_scene_cut] Starting scene cut detection with target_time={target_time}s, fps={fps}, window_seconds={window_seconds}, threshold_hist={threshold_hist}, threshold_pix={threshold_pix}")
    center_frame = int(round(target_time * fps))
    half_window = int(round(window_seconds * fps))
    start_frame = max(0, center_frame - half_window)
    end_frame = min(int(clip.duration * fps) - 1, center_frame + half_window)
    print(f"[find_best_scene_cut] Analyzing frame range: {start_frame} to {end_frame}")

    def get_bgr_and_gray(frame_idx):
 #       print(f"[find_best_scene_cut] Getting frame at index {frame_idx}")
        t = frame_idx / fps
        rgb = clip.get_frame(t)
        img = (rgb * 255).astype(np.uint8)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return bgr, gray

    print("[find_best_scene_cut] Computing histograms and grayscale frames")
    hists = []
    grays = []
    for f in range(start_frame, end_frame + 1):
        bgr, gray = get_bgr_and_gray(f)
        grays.append(gray)
        h = cv2.calcHist([bgr], [0, 1, 2], None, [8, 8, 8], [0, 256]*3)
        cv2.normalize(h, h)
        hists.append(h.flatten())

    print("[find_best_scene_cut] Computing histogram and pixel differences")
    hist_dists = []
    pix_dists = []
    H, W = grays[0].shape
    total_pixels = float(H * W)
    for i in range(len(hists) - 1):
        hd = cv2.compareHist(hists[i], hists[i + 1], cv2.HISTCMP_BHATTACHARYYA)
        hist_dists.append(hd)
        gd = np.sum(np.abs(grays[i + 1].astype(np.int16) - grays[i].astype(np.int16))) / total_pixels
        pix_dists.append(gd / 255.0)
    print(f"[find_best_scene_cut] Computed {len(hist_dists)} histogram and pixel differences")

    center_idx = center_frame - start_frame
    print(f"[find_best_scene_cut] Starting forward scan from index {center_idx}")

    for i in range(center_idx, len(hist_dists)):
        if hist_dists[i] >= threshold_hist and pix_dists[i] >= threshold_pix:
            best_frame = start_frame + i + 1
            print(f"[find_best_scene_cut] Found cut in forward scan at frame {best_frame} (time: {best_frame/fps:.3f}s)")
            return best_frame

    print(f"[find_best_scene_cut] Starting backward scan from index {center_idx - 1}")
    for i in range(center_idx - 1, -1, -1):
        if hist_dists[i] >= threshold_hist and pix_dists[i] >= threshold_pix:
            best_frame = start_frame + i + 1
            print(f"[find_best_scene_cut] Found cut in backward scan at frame {best_frame} (time: {best_frame/fps:.3f}s)")
            return best_frame

    print(f"[find_best_scene_cut] No strong cut found, defaulting to center frame {center_frame} (time: {center_frame/fps:.3f}s)")
    return center_frame

def preliminary_keyword_check(prompt: str) -> bool:
    """Check if the prompt contains video editing-related keywords."""
    print(f"[preliminary_keyword_check] Checking prompt: '{prompt}'")
    editing_keywords = [
        "cut", "trim", "remove", "replace", "join", "merge", "concatenate",
        "video1", "video2", "seconds", "minutes", "start", "end", "middle", "beginning", "last"
    ]
    prompt_lower = prompt.lower()
    has_keywords = any(keyword in prompt_lower for keyword in editing_keywords)
    print(f"[preliminary_keyword_check] Keywords found: {has_keywords}")
    return has_keywords

def validate_video_editing_prompt(prompt: str) -> bool:
    """
    Validate if the prompt is related to video editing using OpenAI.
    """
    print(f"[validate_video_editing_prompt] Validating prompt: /n'{prompt}'")
    if len(prompt.strip()) < 5:
        print("[validate_video_editing_prompt] Prompt too short, invalid")
        return False
    if not client or not OPENAI_API_KEY:
        print("[validate_video_editing_prompt] OpenAI API not configured")
        raise RuntimeError("OpenAI API is not configured")
    try:
        return validate_with_openai(prompt)
    except RuntimeError as e:
        print(f"[validate_video_editing_prompt] Validation error: {str(e)}")
        if "OpenAI API is not configured" in str(e):
            raise HTTPException(503, "Video editing service is temporarily unavailable. Please try again later.")
        else:
            raise HTTPException(500, f"Technical error in validation: {str(e)}")
    except Exception as e:
        print(f"[validate_video_editing_prompt] Unexpected error: {str(e)}")
        raise HTTPException(500, f"Error validating prompt: {str(e)}")

def validate_with_openai(prompt: str) -> bool:
    """Use OpenAI to validate if prompt is a video editing instruction."""
    print("[validate_with_openai] Sending prompt to OpenAI for validation")
    if not preliminary_keyword_check(prompt):
        print("[validate_with_openai] Preliminary check failed: No editing keywords found.")
        return False
    if not client or not OPENAI_API_KEY:
        print("[validate_with_openai] OpenAI API not configured")
        raise RuntimeError("OpenAI API is not configured")
    validation_prompt = f"""
You are an assistant that determines whether a given text is a valid video editing instruction.
A valid video editing instruction must:
1. Refer to one or both videos (e.g., 'video1', 'video2').
2. Specify a clear editing action, such as:
   - Cutting, trimming, or removing parts of a video.
   - Replacing segments of one video with segments from another.
   - Joining or merging videos.
3. If applicable, include specific durations (e.g., '5 seconds') or time positions (e.g., 'start', 'end', 'middle').
Here are examples of valid instructions:
- "Cut the first 5 seconds from video1 and replace with the last 3 seconds of video2."
- "Join video1 and video2 together."
- "Remove the middle 10 seconds from video1."
- "Take the beginning of video2 and put it at the end of video1."
Here are examples of invalid instructions:
- "Make the video look better."
- "Add some music to the video."
- "What's the weather like today?"
- "Cut the video." (too vague, lacks specificity)
Note: The instruction may be phrased in various ways. Focus on the intent: does the user want to manipulate the videos in a way that involves cutting, replacing, or joining segments? If the intent is clear, even if the phrasing is unusual, consider it valid.
Based on these criteria and examples, determine if the following text is a valid video editing instruction:
Text: "{prompt}"
Respond with only "YES" if it is a valid instruction, or "NO" if it is not.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": validation_prompt}],
            max_tokens=10,
            temperature=0
        )
        result = response.choices[0].message.content.strip().upper()
        print(f"[validate_with_openai] OpenAI response: {result}")
        if result not in ["YES", "NO"]:
            print(f"[validate_with_openai] Unexpected response from OpenAI: {result}")
            return False
        return result == "YES"
    except Exception as e:
        print(f"[validate_with_openai] OpenAI API error: {str(e)}")
        raise RuntimeError(f"Failed to validate prompt with OpenAI: {str(e)}")

def parse_prompt_with_openai(prompt: str) -> VideoEditParams:
    """Parse natural language prompt into video edit parameters using direct ChatCompletion with few-shot examples."""
    print(f"[parse_prompt_with_openai] Parsing prompt: '{prompt}'")
    try:
        print("[parse_prompt_with_openai] Validating prompt")
        is_valid = validate_video_editing_prompt(prompt)
    except RuntimeError as e:
        print(f"[parse_prompt_with_openai] Validation error: {str(e)}")
        if "OpenAI API is not configured" in str(e):
            raise HTTPException(503, "Video editing service temporarily unavailable")
        else:
            raise HTTPException(500, f"Validation error: {str(e)}")
    except Exception as e:
        print(f"[parse_prompt_with_openai] Unexpected validation error: {str(e)}")
        raise HTTPException(500, f"Error validating prompt: {str(e)}")

    if not is_valid:
        print("[parse_prompt_with_openai] Prompt is not a valid video editing instruction")
        raise ValueError("Invalid video editing instruction.")

    if not client or not OPENAI_API_KEY:
        print("[parse_prompt_with_openai] OpenAI not available, falling back to simple parsing")
        return parse_prompt_simple(prompt)

    examples = [
        {
            "prompt": "Cut the first 5 seconds from video1 and replace with the last 3 seconds of video2",
            "output": '{"operation": "cut_and_replace", "cut_duration": 5.0, "replace_duration": 3.0, "cut_location": "start", "replace_location": "end"}'
        },
        {
            "prompt": "Join video1 and video2",
            "output": '{"operation": "join", "cut_duration": 0.0, "replace_duration": 0.0, "cut_location": "start", "replace_location": "start"}'
        },
        {
            "prompt": "Trim the last 4 seconds off video2 and substitute with the first 4 seconds of video1",
            "output": '{"operation": "cut_and_replace", "cut_duration": 4.0, "replace_duration": 4.0, "cut_location": "end", "replace_location": "start"}'
        },
        {
            "prompt": "Put the first 5 seconds of video1 before video2",
            "output": '{"operation": "join", "cut_duration": 5.0, "replace_duration": 0.0, "cut_location": "start", "replace_location": "start"}'
        },
        {
            "prompt": "Join 10 seconds of video2 with video1",
            "output": '{"operation": "join", "cut_duration": 10.0, "replace_duration": 0.0, "cut_location": "start", "replace_location": "start"}'
        },
        {
            "prompt": "Chop off the first 5 seconds of video1 and slap on the last 3 seconds of video2",
            "output": '{"operation": "cut_and_replace", "cut_duration": 5.0, "replace_duration": 3.0, "cut_location": "start", "replace_location": "end"}'
        },
        {
            "prompt": "Stick the first 10 seconds of video2 in front of video1",
            "output": '{"operation": "join", "cut_duration": 10.0, "replace_duration": 0.0, "cut_location": "start", "replace_location": "start"}'
        },
        {
            "prompt": "Merge the first 7 seconds of video1 with video2",
            "output": '{"operation": "join", "cut_duration": 7.0, "replace_duration": 0.0, "cut_location": "start", "replace_location": "start"}'
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "You are a video-editing assistant. Given a user instruction, output a JSON object matching the VideoEditParams schema: {operation: str, cut_duration: float, replace_duration: float|null, cut_location: str, replace_location: str}. For 'join' operations with partial clips (e.g., 'join 10 seconds of video2 with video1'), use 'join' with cut_duration set to the specified duration and replace_duration set to 0.0."
        }
    ]
    for example in examples:
        messages.append({"role": "user", "content": example["prompt"]})
        messages.append({"role": "assistant", "content": example["output"]})

    messages.append({"role": "user", "content": prompt})

    try:
        print("[parse_prompt_with_openai] Sending request to OpenAI")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=150,
            temperature=0
        )

        response_text = response.choices[0].message.content.strip()
        print(f"[parse_prompt_with_openai] OpenAI response: {response_text}")

        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                params_dict = json.loads(json_str)
                print(f"[parse_prompt_with_openai] Parsed JSON: {params_dict}")
                return VideoEditParams(**params_dict)
            except json.JSONDecodeError as e:
                print(f"[parse_prompt_with_openai] JSON parsing error: {e}")
                raise ValueError(f"Failed to parse JSON from response: {e}")
        else:
            print("[parse_prompt_with_openai] No JSON found in response")
            raise ValueError("No JSON found in response")

    except Exception as e:
        print(f"[parse_prompt_with_openai] OpenAI parsing failed: {e}, falling back to simple parsing")
        return parse_prompt_simple(prompt)
#If OpenAI not available, use a simple fallback parser
def parse_prompt_simple(prompt: str) -> VideoEditParams:
    """Simple fallback prompt parsing without OpenAI"""
    print(f"[parse_prompt_simple] Using simple parsing for prompt: '{prompt}'")
    prompt_lower = prompt.lower()
    import re
    numbers = re.findall(r'\d+\.?\d*', prompt_lower)
    print(f"[parse_prompt_simple] Extracted numbers: {numbers}")

    if not numbers:
        print("[parse_prompt_simple] No duration found in prompt")
        raise ValueError("No duration found in prompt. Please specify time durations (e.g., '5 seconds', '10 seconds')")

    cut_duration = float(numbers[0])
    replace_duration = float(numbers[1]) if len(numbers) > 1 else cut_duration
    print(f"[parse_prompt_simple] Cut duration: {cut_duration}, Replace duration: {replace_duration}")

    if any(word in prompt_lower for word in ["join", "combine", "merge", "concatenate"]):
        operation = "join"
    else:
        operation = "cut_and_replace"
    print(f"[parse_prompt_simple] Determined operation: {operation}")

    cut_location = "end" if any(word in prompt_lower for word in ["end", "last", "final"]) else "start"
    replace_location = "end" if "end" in prompt_lower and cut_location == "start" else "start"
    print(f"[parse_prompt_simple] Cut location: {cut_location}, Replace location: {replace_location}")

    params = VideoEditParams(
        operation=operation,
        cut_duration=cut_duration,
        replace_duration=replace_duration,
        cut_location=cut_location,
        replace_location=replace_location
    )
    print(f"[parse_prompt_simple] Parsed parameters: {params.dict()}")
    return params

def generate_confirmation_message(params: VideoEditParams) -> str:
    """Generate human-readable confirmation message"""
    print(f"[generate_confirmation_message] Generating message for params: {params.dict()}")
    if params.operation == "join":
        if params.cut_duration > 0:
            message = f"I will take the first {params.cut_duration} seconds from the first video mentioned and join it with the full second video."
        else:
            message = "I will join both videos together into one continuous video."
        print(f"[generate_confirmation_message] Generated message: {message}")
        return message

    replace_duration = params.replace_duration or params.cut_duration
    cut_part = f"first {params.cut_duration} seconds" if params.cut_location == "start" else f"last {params.cut_duration} seconds"
    replace_part = f"first {replace_duration} seconds" if params.replace_location == "start" else f"last {replace_duration} seconds"

    if params.cut_location == "start":
        message = f"I will remove the {cut_part} from Video 1 and replace it with the {replace_part} from Video 2 at the beginning."
    else:
        message = f"I will remove the {cut_part} from Video 1 and replace it with the {replace_part} from Video 2 at the end."
    print(f"[generate_confirmation_message] Generated message: {message}")
    return message

def match_video_properties(source_clip, reference_clip):
    """Match video properties without causing corruption"""
    print(f"[match_video_properties] Matching properties of source clip to reference clip")
    try:
        # Resize to match dimensions if different
        if source_clip.size != reference_clip.size:
            print(f"[match_video_properties] Resizing source clip from {source_clip.size} to {reference_clip.size}")
            source_clip = source_clip.resize(reference_clip.size)

        # Match fps if different
        if abs(source_clip.fps - reference_clip.fps) > 0.1:
            print(f"[match_video_properties] Setting FPS from {source_clip.fps} to {reference_clip.fps}")
            source_clip = source_clip.set_fps(reference_clip.fps)

        return source_clip
    except Exception as e:
        print(f"[match_video_properties] Error matching properties: {str(e)}")
        return source_clip

def safe_clip_processing(clip1, clip2, params: VideoEditParams):
    """Safely process video clips with error handling"""
    print(f"[safe_clip_processing] Processing clips with params: {params.dict()}")
    try:
        replace_duration = params.replace_duration or params.cut_duration
        extra_duration = 0.5  # Add 0.5 seconds of extra frames after the cut point
        print(f"[safe_clip_processing] Replace duration set to: {replace_duration}s, extra_duration: {extra_duration}s")

        if params.cut_duration > clip1.duration:
            print(f"[safe_clip_processing] Error: Cut duration {params.cut_duration}s exceeds video1 length {clip1.duration:.1f}s")
            raise ValueError(f"Cut duration {params.cut_duration}s exceeds video length {clip1.duration:.1f}s")
        if replace_duration > clip2.duration:
            print(f"[safe_clip_processing] Error: Replace duration {replace_duration}s exceeds video2 length {clip2.duration:.1f}s")
            raise ValueError(f"Replace duration {replace_duration}s exceeds video length {clip2.duration:.1f}s")

        fps = int(clip1.fps)
        print(f"[safe_clip_processing] Clip1 duration: {clip1.duration}s, FPS: {fps}")
        print(f"[safe_clip_processing] Clip2 duration: {clip2.duration}s")

        # Process first clip
        if params.operation == "cut_and_replace":
            if params.cut_location == "start":
                print(f"[safe_clip_processing] Finding best cut for video1 at start, target time: {params.cut_duration}s")
                best_frame = find_best_scene_cut(clip1, params.cut_duration, fps)
                t_cut = best_frame / fps
                clip1_edited = clip1.subclip(t_cut)
                print(f"[safe_clip_processing] Cut video1 from {t_cut:.3f}s to end")
            else:
                nominal_cut_time = clip1.duration - params.cut_duration
                print(f"[safe_clip_processing] Finding best cut for video1 at end, target time: {nominal_cut_time}s")
                best_frame = find_best_scene_cut(clip1, nominal_cut_time, fps)
                t_end = best_frame / fps
                clip1_edited = clip1.subclip(0, t_end)
                print(f"[safe_clip_processing] Cut video1 from start to {t_end:.3f}s")

            # Process second clip for replacement
            if params.replace_location == "start":
                print(f"[safe_clip_processing] Finding best cut for video2 at start, target time: {replace_duration}s")
                best_frame = find_best_scene_cut(clip2, replace_duration, fps)
                t_end = best_frame / fps
                t_start = max(0, t_end - replace_duration - extra_duration)
                t_end = min(clip2.duration, t_end + extra_duration)
                clip2_segment = clip2.subclip(t_start, t_end)
                print(f"[safe_clip_processing] Cut video2 from {t_start:.3f}s to {t_end:.3f}s")
            else:
                nominal_cut_time = clip2.duration - replace_duration
                print(f"[safe_clip_processing] Finding best cut for video2 at end, target time: {nominal_cut_time}s")
                best_frame = find_best_scene_cut(clip2, nominal_cut_time, fps)
                t_end = best_frame / fps
                t_start = max(0, t_end - replace_duration - extra_duration)
                t_end = min(clip2.duration, t_end + extra_duration)
                clip2_segment = clip2.subclip(t_start, t_end)
                print(f"[safe_clip_processing] Cut video2 from {t_start:.3f}s to {t_end:.3f}s")

            # Join clips
            if params.cut_location == "start":
                clips_to_join = [clip2_segment, clip1_edited]
                print("[safe_clip_processing] Joining clips: [video2_segment, video1_edited]")
            else:
                clips_to_join = [clip1_edited, clip2_segment]
                print("[safe_clip_processing] Joining clips: [video1_edited, video2_segment]")
            final_clip = concatenate_videoclips(clips_to_join, method="compose")
            print("[safe_clip_processing] Performed cut and replace operation")

        elif params.operation == "join":
            if params.cut_duration > 0:
                # For join with a duration, assume it's the duration of the first video mentioned
                print(f"[safe_clip_processing] Joining first {params.cut_duration}s of first video with full second video")
                # Assume clip2 provides the partial clip
                if params.cut_duration > clip2.duration:
                    print(f"[safe_clip_processing] Error: Cut duration {params.cut_duration}s exceeds video2 length {clip2.duration:.1f}s")
                    raise ValueError(f"Cut duration exceeds video2 length")
                best_frame = find_best_scene_cut(clip2, params.cut_duration, fps)
                t_end = best_frame / fps
                clip2_segment = clip2.subclip(0, t_end)
                clip2_segment = match_video_properties(clip2_segment, clip1)
                clips_to_join = [clip2_segment, clip1]
                print(f"[safe_clip_processing] Joining clips: [video2_segment ({t_end:.3f}s), video1]")
            else:
                clip2_matched = match_video_properties(clip2, clip1)
                clips_to_join = [clip1, clip2_matched]
                print(f"[safe_clip_processing] Joining full clips: [video1, video2]")
            final_clip = concatenate_videoclips(clips_to_join, method="compose")
            print("[safe_clip_processing] Joined clips")

        else:
            print(f"[safe_clip_processing] Error: Unknown operation {params.operation}")
            raise ValueError(f"Unknown operation: {params.operation}")

        final_clip = final_clip.set_fps(clip1.fps)
        print("[safe_clip_processing] Final clip created with FPS:", clip1.fps)
        return final_clip

    except Exception as e:
        print(f"[safe_clip_processing] Video processing failed: {str(e)}")
        raise ValueError(f"Video processing failed: {str(e)}")

async def cleanup_temp_files(file_paths: list):
    """Background task to cleanup temporary files"""
    print(f"[cleanup_temp_files] Scheduling cleanup for files: {file_paths}")
    await asyncio.sleep(5)
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                print(f"[cleanup_temp_files] Deleted file: {file_path}")
            except Exception as e:
                print(f"[cleanup_temp_files] Error deleting file {file_path}: {str(e)}")

# Global variable to store the last parsed parameters
last_parsed_params = None

@app.post("/edit-video")
async def edit_video(
    background_tasks: BackgroundTasks,
    video1: UploadFile = File(...),
    video2: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    New flow:
    - The first request: user sends editing prompt. We parse it, store params, return confirmation plan.
    - The second request: user sends just "yes" or "no". 
      If "yes", we use stored params to do the editing.
      If "no", we cancel and reset.
    """
    global last_parsed_params
    print(f"[edit_video] Received request with prompt: '{prompt}'")
    print(f"[edit_video] Video1 filename: {video1.filename}, Video2 filename: {video2.filename}")

    prompt_clean = prompt.strip().lower()
    print(f"[edit_video] Cleaned prompt: '{prompt_clean}'")

    if prompt_clean == "yes":
        print("[edit_video] User confirmed with 'yes'")
        if last_parsed_params is None:
            print("[edit_video] Error: No previous editing plan found")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No previous editing plan found. Please send an editing prompt first."
                }
            )

        params = last_parsed_params
        print(f"[edit_video] Using stored parameters: {params.dict()}")
        
        for video, name in [(video1, "video1"), (video2, "video2")]:
            if not video.filename or not video.filename.lower().endswith('.mp4'):
                print(f"[edit_video] Error: Invalid file type for {name}")
                raise HTTPException(400, f"{name} must be an MP4 file")

        try:
            video1_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            video2_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            print(f"[edit_video] Saving video1 to {video1_path}")
            print(f"[edit_video] Saving video2 to {video2_path}")

            with open(video1_path, 'wb') as f1:
                f1.write(await video1.read())
            with open(video2_path, 'wb') as f2:
                f2.write(await video2.read())
            print("[edit_video] Videos saved to temporary files")

            print("[edit_video] Loading video clips")
            clip1 = VideoFileClip(video1_path)
            clip2 = VideoFileClip(video2_path)
            print("[edit_video] Clips loaded successfully")

            final_clip = safe_clip_processing(clip1, clip2, params)

            output_path = os.path.join(tempfile.gettempdir(), f"edited_{uuid.uuid4().hex}.mp4")
            print(f"[edit_video] Writing output to {output_path}")
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=clip1.fps,
                verbose=False,
                logger=None,
                preset='medium',
                ffmpeg_params=['-crf', '23']
            )
            print("[edit_video] Output video written")

            for clip in [clip1, clip2, final_clip]:
                if clip:
                    clip.close()
            print("[edit_video] Closed all video clips")

            print("[edit_video] Scheduling cleanup of temporary files")
            background_tasks.add_task(cleanup_temp_files, [video1_path, video2_path, output_path])

            if not os.path.exists(output_path):
                print("[edit_video] Error: Output file was not created")
                raise HTTPException(500, "Output file was not created")

            last_parsed_params = None
            print("[edit_video] Reset last_parsed_params")

            print("[edit_video] Returning edited video")
            return FileResponse(
                output_path,
                media_type="video/mp4",
                filename=f"edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )

        except Exception as e:
            last_parsed_params = None
            print(f"[edit_video] Video processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

    elif prompt_clean == "no":
        print("[edit_video] User canceled with 'no'")
        last_parsed_params = None
        print("[edit_video] Reset last_parsed_params")
        return JSONResponse(
            status_code=200,
            content={
                "status": "rejected",
                "message": "Edit canceled by user."
            }
        )

    try:
        params = parse_prompt_with_openai(prompt.strip())
        print(f"[edit_video] Parsed parameters: {params.dict()}")
        
        last_parsed_params = params
        print("[edit_video] Stored parameters globally")

        confirmation_message = generate_confirmation_message(params)
        print(f"[edit_video] Generated confirmation message: {confirmation_message}")

        return JSONResponse(
            status_code=200,
            content={
                "status": "confirmation_required",
                "plan": confirmation_message,
                "details": {
                    "operation": params.operation,
                    "cut_duration": params.cut_duration,
                    "replace_duration": params.replace_duration,
                    "cut_location": params.cut_location,
                    "replace_location": params.replace_location
                },
                "next_step": "Send 'yes' to confirm or 'no' to cancel."
            }
        )

    except Exception as e:
        print(f"[edit_video] Failed to understand editing prompt: {str(e)}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": f"Failed to understand editing prompt: {str(e)}",
                "examples": [
                    "Cut first 5 seconds from video1 and replace with last 3 seconds of video2",
                    "Join both videos together"
                ]
            }
        )

@app.get("/")
async def root():
    print("[root] Received GET request to root endpoint")
    return {
        "message": "Professional Video Editor API v2.0",
        "endpoint": "/edit-video",
        "workflow": [
            "1. POST /edit-video with video1, video2, and prompt (no 'yes' or 'no').",
            "2. Server returns a confirmation message (plan + details).",
            "3. Client resends the same prompt but appends 'yes' to confirm (or 'no' to cancel).",
            "4. If 'yes', download the edited video; if 'no', operation is canceled."
        ]
    }

@app.on_event("startup")
async def startup_event():
    print("Video Editor API Starting...")
    print("Endpoint: /edit-video")
    print("Make sure OPENAI_API_KEY is set in .env file")

if __name__ == "__main__":
    print("[main] Starting Uvicorn server")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)