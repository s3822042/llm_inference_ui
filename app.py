import os
import streamlit as st
import whisper
from chroma_init import ChromaProcessor
from audio_llm_inference import AudioTextTranscriptionAnalysis

ALLOWED_FILE_TYPES = {
    "Audio Files": ["mp3", "wav", "flac"]
}
ALLOW_MULTIPLE_FILES = True
FILE_STORAGE_PATH = "./files"

model = None
chroma = ChromaProcessor()
at = AudioTextTranscriptionAnalysis()
def main():
    st.title("Audio File Upload and Query App")
    st.write("Upload an audio file (e.g., MP3, WAV) and enter a query to see the results.")

    duma = False

    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Choose a file", 
            type=ALLOWED_FILE_TYPES['Audio Files'],
            accept_multiple_files=ALLOW_MULTIPLE_FILES
            )
        submitted = st.form_submit_button("submit")    

    if duma == False:        
        on_upload(uploaded_files)        
        duma = True
        
    query = st.text_input("Enter your query")

    results_placeholder = st.empty()

    if st.button("Submit"):
        if uploaded_files is not None and query:
            results = process_query(query)
            st.write(results)
            # display_results(results, results_placeholder)
        else:
            st.warning("Please upload an audio file and enter a query.")

    apply_styles()

def on_upload(uploaded_files):
    for uploaded_file in uploaded_files:
        upload_file(uploaded_file.name, uploaded_file.read())
        transcription = transcript_file(uploaded_file.name)
        ingest_into_db(transcription)

def upload_file(name, bytes):
    file_path = f'{FILE_STORAGE_PATH}/{name}'
    
    # convert file path to file object

    if (os.path.exists(file_path)):
        st.warning(f'File {name} already exists.')
        return
    
    with open(file_path, 'wb') as f:
        f.write(bytes)
        st.success(f'File {name} successfully uploaded.')

def transcript_file(name):
    file_path = f'{FILE_STORAGE_PATH}/{name}'
    data_load_state = st.text(f'transcribing {file_path}...')
    options = {
        "language": "English"
    }
    model = whisper.load_model('base')
    result = model.transcribe(
        file_path,
        **options
    )
    print(result)
    data_load_state.text('transcribing...done!')
    return result

def ingest_into_db(transcription):
    # st.write()
    output_file_path = "tmp.txt"
    with open(output_file_path, "r+") as output_file:
        if output_file.read():
            output_file.seek(0)
            output_file.truncate()
        print('----Writing to file')
    # Write the extracted content to the text file
    with open(output_file_path, "w") as output_file:
        output_file.write(transcription['text'])
    print(f"Extracted content has been written to {output_file_path}")
    chroma.process_and_persist('tmp.txt')

def process_query(query):
    results =  at.transcript_qa(chroma, query)
    
    return results

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

if __name__ == "__main__":
    main()
