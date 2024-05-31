import logging
import os
from pathlib import Path
import shutil

import gradio as gr

from decode import decode
from model import get_pretrained_model, get_vad, language_to_models, get_punct_model
import subprocess

title = "video to text"

css = """
.result {display:flex;flex-direction:column}
.result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
.result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
.result_item_error {background-color:#ff7070;color:white;align-self:start}
"""

project_dir = Path(__file__).parent.resolve()
upload_dir = project_dir / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)

def update_model_dropdown(language: str):
    if language in language_to_models:
        choices = language_to_models[language]
        return gr.Dropdown(
            choices=choices,
            value=choices[0],
            interactive=True,
        )
    raise ValueError(f"Unsupported language: {language}")

def build_html_output(s: str, style: str = "result_item_success"):
    return f"""
    <div class='result'>
        <div class='result_item {style}'>
          {s}
        </div>
    </div>
    """

def show_file_info(in_filename: str):
    logging.info(f"Input file: {in_filename}")
    _ = os.system(f"ffprobe -hide_banner -i \"{in_filename}\"")

def process_uploaded_video_file(
    language: str,
    repo_id: str,
    add_punctuation: str,
    in_filename: str,
):
    if in_filename is None or in_filename == "":
        return (
            "",
            build_html_output(
                "Please first upload a file and then click "
                'the button "submit for recognition"',
                "result_item_error",
            ),
            "",
            "",
        )

    logging.info(f"Processing uploaded video file: {in_filename}")

    ans = process(language, repo_id, add_punctuation, in_filename)
    return (in_filename, ans[0]), ans[0], ans[1], ans[2], ans[3]

def process_uploaded_audio_file(
    language: str,
    repo_id: str,
    add_punctuation: str,
    in_filename: str,
):
    if in_filename is None or in_filename == "":
        return (
            "",
            build_html_output(
                "Please first upload a file and then click "
                'the button "submit for recognition"',
                "result_item_error",
            ),
            "",
            "",
        )

    logging.info(f"Processing uploaded audio file: {in_filename}")

    return process(language, repo_id, add_punctuation, in_filename)

def process(language: str, repo_id: str, add_punctuation: str, in_filename: str):
    logging.info(f"add_punctuation: {add_punctuation}")
    recognizer = get_pretrained_model(repo_id)
    vad = get_vad()

    if "whisper" in repo_id:
        add_punctuation = "No"

    if add_punctuation == "Yes":
        punct = get_punct_model()
    else:
        punct = None

    result, all_text = decode(recognizer, vad, punct, in_filename)
    logging.info(result)

    srt_filename = Path(in_filename).with_suffix(".srt")
    with open(srt_filename, "w", encoding="utf-8") as f:
        f.write(result)

    show_file_info(in_filename)
    logging.info(f"all_text:\n{all_text}")
    logging.info("Done")

    return (
        str(srt_filename),
        build_html_output("Done! Please download the SRT file", "result_item_success"),
        result,
        all_text,
    )

def cleanup_uploads_folder(folder_path):
    """Deletes all files in the specified folder."""
    folder = Path(folder_path)
    for file_path in folder.iterdir():
        if file_path.is_file():  
            file_path.unlink()

def combine_subtitles_with_video(video_file: str, subtitle_file: str):
    logging.info(f"Combining subtitles from {subtitle_file} with video {video_file}")

    combined_video_filename = Path("uploads/output_video.mp4")
    
    command = 'ffmpeg -i "uploads/uploaded_video.mp4" -vf "subtitles=uploads/uploaded_subtitles.srt" "uploads/output_video.mp4"'

    try:
        subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logging.info("Subtitles combined with video successfully!")
        html_output = build_html_output("Subtitles combined with video successfully!", "result_item_success")
        return str(combined_video_filename), html_output, str(combined_video_filename)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        html_output = build_html_output(f"Failed to combine subtitles with video. FFmpeg error: {e.stderr}", "result_item_error")
        return "", html_output, ""

def save_uploaded_file(file, filename):
    file_path = upload_dir / filename
    shutil.move(file.name, file_path)
    return file_path

demo = gr.Blocks(css=css)

with demo:
    gr.Markdown(title)
    language_choices = list(language_to_models.keys())

    language_radio = gr.Radio(
        label="Language",
        choices=language_choices,
        value=language_choices[0],
    )

    model_dropdown = gr.Dropdown(
        choices=language_to_models[language_choices[0]],
        label="Select a model",
        value=language_to_models[language_choices[0]][0],
    )

    language_radio.change(
        update_model_dropdown,
        inputs=language_radio,
        outputs=model_dropdown,
    )
    punct_radio = gr.Radio(
        label="Whether to add punctuation",
        choices=["Yes", "No"],
        value="Yes",
    )

    with gr.Tabs():
        with gr.TabItem("Upload video from disk"):
            uploaded_video_file = gr.Video(
                sources=["upload"],
                label="Upload from disk",
                show_share_button=True,
            )
            upload_video_button = gr.Button("Submit for recognition")

            output_video = gr.Video(label="Output")
            output_srt_file_video = gr.File(
                label="Generated subtitles", show_label=True
            )

            output_info_video = gr.HTML(label="Info")
            output_textbox_video = gr.Textbox(
                label="Recognized speech from uploaded video file (srt format)"
            )
            all_output_textbox_video = gr.Textbox(
                label="Recognized speech from uploaded video file (all in one)"
            )

        with gr.TabItem("Upload audio from disk"):
            uploaded_audio_file = gr.Audio(
                sources=["upload"],  
                type="filepath",
                label="Upload audio from disk",
            )
            upload_audio_button = gr.Button("Submit for recognition")

            output_srt_file_audio = gr.File(
                label="Generated subtitles", show_label=True
            )

            output_info_audio = gr.HTML(label="Info")
            output_textbox_audio = gr.Textbox(
                label="Recognized speech from uploaded audio file (srt format)"
            )
            all_output_textbox_audio = gr.Textbox(
                label="Recognized speech from uploaded audio file (all in one)"
            )

        with gr.TabItem("Upload two files"):
            file_input1 = gr.File(label="Upload first file (MP4 only)", file_types=["mp4"])
            file_input2 = gr.File(label="Upload second file (Text only)", file_types=["srt"])
            process_files_button = gr.Button("Process files")

            output_combined_video = gr.Video(label="Combined video with subtitles", show_label=True)
            output_info_combined_video = gr.HTML(label="Info for combined video")

        upload_video_button.click(
            process_uploaded_video_file,
            inputs=[
                language_radio,
                model_dropdown,
                punct_radio,
                uploaded_video_file,
            ],
            outputs=[
                output_video,
                output_srt_file_video,
                output_info_video,
                output_textbox_video,
                all_output_textbox_video,
            ],
        )

        upload_audio_button.click(
            process_uploaded_audio_file,
            inputs=[
                language_radio,
                model_dropdown,
                punct_radio,
                uploaded_audio_file,
            ],
            outputs=[
                output_srt_file_audio,
                output_info_audio,
                output_textbox_audio,
                all_output_textbox_audio,
            ],
        )

        def process_files(language, repo_id, add_punctuation, file1, file2):
            logging.info(f"Processing files: {file1}, {file2}")

            video_path = save_uploaded_file(file1, "uploaded_video.mp4")
            subtitle_text_path = save_uploaded_file(file2, "uploaded_subtitles.txt")

            subtitle_srt_path = subtitle_text_path.with_suffix(".srt")
            subtitle_text_path.rename(subtitle_srt_path)

            combined_video_filename, html_output, combined_output = combine_subtitles_with_video(video_path, subtitle_srt_path)

            return (
                combined_video_filename,
                html_output,
                combined_output,
            )
        
        process_files_button.click(
            process_files,
            inputs=[
                language_radio,
                model_dropdown,
                punct_radio,
                file_input1,
                file_input2,
            ],
            outputs=[
                output_combined_video,
                output_info_combined_video,
                output_info_combined_video,
            ],
        )

if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    demo.launch()
