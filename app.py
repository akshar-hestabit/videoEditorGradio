import gradio as gr
import requests
import tempfile
import os
from typing import Optional, Tuple
import subprocess
import threading
# FastAPI backend URL

def run_fastapi():
    subprocess.run(["uvicorn", "video_m:app", "--host", "0.0.0.0", "--port", "8002"])

threading.Thread(target=run_fastapi).start()
API_URL = "http://127.0.0.1:8002/"

def process_video_request(video1, video2, prompt, confirmation_step) -> Tuple[str, str, bool, Optional[str]]:
    """
    Process video editing request through the FastAPI backend.
    Returns: (message, confirmation_text, show_confirmation, video_path)
    """
    if not video1 or not video2:
        return "Please upload both videos.", "", False, None
    
    if not prompt.strip():
        return "Please enter an editing prompt.", "", False, None
    
    try:
        # Prepare files for API request
        with open(video1, 'rb') as f1, open(video2, 'rb') as f2:
            files = {
                'video1': ('video1.mp4', f1, 'video/mp4'),
                'video2': ('video2.mp4', f2, 'video/mp4')
            }
            data = {'prompt': prompt.strip()}
            
            response = requests.post(f"{API_URL}/edit-video", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("status") == "confirmation_required":
                # First step: Show confirmation
                plan = result.get("plan", "")
                return f"Editing Plan:\n{plan}", plan, True, None
            elif result.get("status") == "rejected":
                # User said no
                return "Edit canceled by user.", "", False, None
        
        elif response.headers.get('content-type') == 'video/mp4':
            # Second step: Video returned
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_video.write(response.content)
            temp_video.close()
            return "Video edited successfully!", "", False, temp_video.name
        
        else:
            # Error response
            try:
                error_data = response.json()
                return f"Error: {error_data.get('message', 'Unknown error')}", "", False, None
            except:
                return f"Error: HTTP {response.status_code}", "", False, None
                
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to video editing service. Make sure the FastAPI server is running on localhost:8000", "", False, None
    except Exception as e:
        return f"Error: {str(e)}", "", False, None

def confirm_edit(video1, video2, confirmation_text, accept: bool) -> Tuple[str, bool, Optional[str]]:
    """
    Handle user confirmation (yes/no) and process the edit.
    Returns: (message, show_confirmation, video_path)
    """
    if not video1 or not video2 or not confirmation_text:
        return "Missing information for confirmation.", False, None
    
    try:
        prompt = "yes" if accept else "no"
        
        with open(video1, 'rb') as f1, open(video2, 'rb') as f2:
            files = {
                'video1': ('video1.mp4', f1, 'video/mp4'),
                'video2': ('video2.mp4', f2, 'video/mp4')
            }
            data = {'prompt': prompt}
            
            response = requests.post(f"{API_URL}/edit-video", files=files, data=data)
        
        if response.headers.get('content-type') == 'video/mp4':
            # Success: Video returned
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_video.write(response.content)
            temp_video.close()
            return "Video edited successfully! Download your edited video below.", False, temp_video.name
        
        elif response.status_code == 200:
            result = response.json()
            if result.get("status") == "rejected":
                return "Edit canceled.", False, None
        
        # Error cases
        try:
            error_data = response.json()
            return f"Error: {error_data.get('message', 'Unknown error')}", False, None
        except:
            return f"Error: HTTP {response.status_code}", False, None
            
    except Exception as e:
        return f"Error: {str(e)}", False, None

# Create Gradio interface
with gr.Blocks(title="Video Editor", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎬 AI Video Editor")
    gr.Markdown("Upload two videos and describe how you want to edit them!")
    
    with gr.Row():
        with gr.Column():
            video1_input = gr.File(label="Video 1", file_types=[".mp4"])
            video2_input = gr.File(label="Video 2", file_types=[".mp4"])
            
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Editing Prompt", 
                placeholder="e.g., 'Cut first 5 seconds from video1 and replace with last 3 seconds of video2'",
                lines=3
            )
            submit_btn = gr.Button("🎯 Create Editing Plan", variant="primary")
    
    # Status and confirmation section
    status_text = gr.Textbox(label="Status", interactive=False, lines=3)
    
    # Confirmation section (hidden initially)
    with gr.Group(visible=False) as confirmation_group:
        gr.Markdown("### Confirm Your Edit")
        confirmation_text = gr.Textbox(label="Editing Plan", interactive=False, lines=2)
        
        with gr.Row():
            confirm_yes = gr.Button("✅ Yes, Edit Video", variant="primary")
            confirm_no = gr.Button("❌ No, Cancel", variant="secondary")
    
    # Output section
    output_video = gr.File(label="📥 Download Edited Video", visible=False)
    
    # Hidden state to store confirmation text
    confirmation_state = gr.State("")
    
    # Event handlers
    def handle_submit(video1, video2, prompt):
        message, conf_text, show_conf, video_path = process_video_request(video1, video2, prompt, False)
        
        return (
            message,  # status_text
            gr.update(visible=show_conf),  # confirmation_group
            conf_text,  # confirmation_text
            conf_text,  # confirmation_state
            gr.update(value=video_path, visible=video_path is not None)  # output_video
        )
    
    def handle_confirmation(video1, video2, conf_state, accept):
        message, show_conf, video_path = confirm_edit(video1, video2, conf_state, accept)
        
        return (
            message,  # status_text
            gr.update(visible=False),  # confirmation_group (hide after decision)
            gr.update(value=video_path, visible=video_path is not None)  # output_video
        )
    
    # Connect events
    submit_btn.click(
        handle_submit,
        inputs=[video1_input, video2_input, prompt_input],
        outputs=[status_text, confirmation_group, confirmation_text, confirmation_state, output_video]
    )
    
    confirm_yes.click(
        lambda v1, v2, conf: handle_confirmation(v1, v2, conf, True),
        inputs=[video1_input, video2_input, confirmation_state],
        outputs=[status_text, confirmation_group, output_video]
    )
    
    confirm_no.click(
        lambda v1, v2, conf: handle_confirmation(v1, v2, conf, False),
        inputs=[video1_input, video2_input, confirmation_state],
        outputs=[status_text, confirmation_group, output_video]
    )
    
    # Examples
    gr.Examples(
        examples=[
            ["Cut the first 5 seconds from video1 and replace with the last 3 seconds of video2"],
            ["Join video1 and video2 together"],
            ["Remove 3 seconds from the middle of video1 and replace with first 3 seconds of video2"],
            ["Take off 2 seconds from video1 from middle and replace with beginning 2 seconds of video2"]
        ],
        inputs=[prompt_input]
    )

if __name__ == "__main__":
    print("Starting Gradio Video Editor...")
    print(":")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)