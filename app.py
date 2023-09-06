import os
import glob
import streamlit as st
import whisper
from chroma_init import ChromaProcessor
from audio_llm_inference import AudioTextTranscriptionAnalysis

ALLOWED_FILE_TYPES = {
    "Audio Files": ["mp3", "wav", "flac", "m4a"]
}
ALLOW_MULTIPLE_FILES = True
FILE_STORAGE_PATH = "./files"

model = None
chroma = ChromaProcessor()
at = AudioTextTranscriptionAnalysis()

def main():
    st.title("Audio File Upload and Query App")
    st.write("Upload an audio file (e.g., MP3, WAV, M4A) and enter a query to see the results.")

    isUploading = False

    with st.form("upload-form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Choose a file",
            type=ALLOWED_FILE_TYPES['Audio Files'],
            accept_multiple_files=ALLOW_MULTIPLE_FILES
            )
        submitted = st.form_submit_button("submit")

        if submitted and uploaded_files is not None and not isUploading:
            on_upload(uploaded_files)
            isUploading = True

        if uploaded_files:
            display_uploaded_files(uploaded_files)

    with st.form('query-form', clear_on_submit=True):
        query = st.text_input("Enter your query")

        submitted = st.form_submit_button("submit")

        if submitted and query:
            # st.button("Submit", key=submit_button_key, disabled=True)
            results = process_query(query)
            st.code(query)
            st.markdown(results)
            # display_results(results, results_placeholder)

    apply_styles()

def on_upload(uploaded_files):
    uploaded_file_paths = []
    uploaded_file_names = []
    for uploaded_file in uploaded_files:
        upload_file(uploaded_file.name, uploaded_file.read())
        uploaded_file_paths.append(f'{FILE_STORAGE_PATH}/{uploaded_file.name}')
        uploaded_file_names.append(uploaded_file.name)

    transcriptions = transcript_files(uploaded_file_paths)
    ingest_into_db(transcriptions, uploaded_file_names)

def upload_file(name, bytes):
    file_path = f'{FILE_STORAGE_PATH}/{name}'

    # convert file path to file object
    if (os.path.exists(file_path)):
        st.warning(f'File {name} already exists.')
        return

    with open(file_path, 'wb') as f:
        f.write(bytes)
        st.success(f'File {name} successfully uploaded.')

def transcript_files(file_paths):
    data_load_state = st.text("Transcribing files...")
    options = {
        "language": "English"
    }
    model = whisper.load_model('base')
    results = []
    for file_path in file_paths:
        result = model.transcribe(
            file_path,
            **options
        )
        results.append(result)
    data_load_state.text('Transcribing files...done!')
    return results

def ingest_into_db(transcriptions, file_names):
    concatenated_transcription = ""

    for transcription, file_name in zip(transcriptions, file_names):
        concatenated_transcription += transcription['text'] + "\n"

        individual_file_path = f'{FILE_STORAGE_PATH}/{file_name.replace(".", "_")}_transcript.txt'
        with open(individual_file_path, "w") as output_file:
            output_file.write(transcription['text'])

        print(f"Transcript for {file_name} has been written to {individual_file_path}")

    concatenated_file_path = f'{FILE_STORAGE_PATH}/all_transcriptions.txt'
    with open(concatenated_file_path, "w") as output_file:
        output_file.write(concatenated_transcription)

    print("Concatenated transcription has been written to", concatenated_file_path)

    # chroma.process_and_persist(concatenated_file_path)
    chroma.load_document(concatenated_file_path)


def process_query(query):
    results =  at.transcript_qa(chroma, query)
    return results

def display_uploaded_files(uploaded_files):
    st.markdown("### Uploaded Files:")
    for uploaded_file in uploaded_files:
        st.write(f"- {uploaded_file.name}")
        transcript_link = create_transcript_link(uploaded_file.name)
        st.write(transcript_link)

def create_transcript_link(file_name):
    transcript_path = f'{FILE_STORAGE_PATH}/{file_name.replace(".", "_")}_transcript.txt'
    if os.path.exists(transcript_path):
        return f"[Download Transcript](/{transcript_path})"
    else:
        return "Transcript not available"

def display_results(results, results_placeholder):
    results_placeholder.markdown("### Results:")
    for idx, result in enumerate(results, start=1):
        results_placeholder.write(f"{idx}. {result}")

def apply_styles():
    st.markdown(
        """
        <style>
        .stButton button {
            background-color: #007BFF !important;
            color: white !important;
            font-weight: bold;
            border-radius: 5px;
        }
        .stFileUploader>div>div>div {
            background-color: #f0f0f0;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def cleanUpFiles():
    files = glob.glob('{FILE_STORAGE_PATH}/*')
    for f in files:
        os.remove(f)

if __name__ == "__main__":
    cleanUpFiles()
    main()
