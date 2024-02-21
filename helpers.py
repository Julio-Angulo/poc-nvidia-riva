import openai
from openai import OpenAI
from pdfminer.high_level import extract_text
import re
from singlestoredb import create_engine
from sqlalchemy import text

import io
import IPython.display as ipd
import grpc
import riva.client
import numpy as np
from  scipy.io import wavfile
import calendar 
import time
import datetime
import os

client = OpenAI()


def print_pdf_text(url=None, file_path=None):
    # Determine the source of the PDF (URL or local file)
    if url:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        temp_file_path = "temp_pdf_file.pdf"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(response.content)  # Save the PDF to a temporary file
        pdf_source = temp_file_path
    elif file_path:
        pdf_source = file_path  # Set the source to the provided local file path
    else:
        raise ValueError("Either url or file_path must be provided.")

    # Extract text using pdfminer
    text = extract_text(pdf_source)

    # Remove special characters except "@", "+", ".", and "/"
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s@+./:,]", "", text)

    # Format the text for better readability
    cleaned_text = cleaned_text.replace("\n\n", " ").replace("\n", " ")
    # If a temporary file was used, delete it
    if url and os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return cleaned_text

def pinfo_extractor(resume_text):
    context = f"Resume text: {resume_text}"
    question = """ From above candidate's resume text, extract the only following details:
                Name: (Find the candidate's full name. If not available, specify "not available.")
                Email: (Locate the candidate's email address. If not available, specify "not available.")
                Phone Number: (Identify the candidate's phone number. If not found, specify "not available.")
                Years of Experience: (If not explicitly mentioned, calculate the years of experience by analyzing the time durations at each company or position listed. Sum up the total durations to estimate the years of experience. If not determinable, write "not available.")
                Skills Set: Extract the skills which are purely technical and represent them as: [skill1, skill2,... <other skills from resume>]. If no skills are provided, state "not available."
                Technologies Set: Extract the technologies which are purely technical and represent them as: [technology1, technology2,... <other technologies from resume>]. If no technologies are provided, state "not available."
                Profile: (Identify the candidate's job profile or designation. If not mentioned, specify "not available.")
                Summary: provide a brief summary of the candidate's profile without using more than one newline to segregate sections.
                """

    prompt = f"""
        Based on the below given candidate information, only answer asked question:
        {context}
        Question: {question}
    """
    # print(prompt)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful HR recruiter."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
        temperature=0.5,
        n=1,  # assuming you want one generation per document
    )
    # Extract the generated response
    response_text = response.choices[
        0
    ].message.content  # response['choices'][0]['message']['content']
    # print(response_text)
    # Split the response_text into lines
    lines = response_text.strip().split("\n")

    # Now, split each line on the colon to separate the labels from the values
    # Extract the values
    name = lines[0].split(": ")[1]
    email = lines[1].split(": ")[1]
    phone_no = lines[2].split(": ")[1]
    years_of_experience = lines[3].split(": ")[1]
    skills = lines[4].split(": ")[1]
    technologies = lines[5].split(": ")[1]
    profile = lines[6].split(": ")[1]
    summary = lines[7].split(": ")[1]
    data_dict = {
        "name": name,
        "email": email,
        "phone_no": phone_no,
        "years_of_experience": years_of_experience,
        "skills": skills,
        "technologies": technologies,
        "profile": profile,
        "summary": summary,
    }
    # print(data_dict, "\n")
    return data_dict

def add_data_to_db(input_dict):
    # Create the SQLAlchemy engine
    engine = create_engine("mysql://root:password@localhost:3306/resume_evaluator", future=True)

    # Create the SQL query for inserting the data
    query_sql = f"""
        INSERT INTO resumes_profile_data (names, email, phone_no, years_of_experience, skills, technologies, profile_name, resume_summary)
        VALUES ("{input_dict['name']}", "{input_dict['email']}", "{input_dict['phone_no']}", "{input_dict['years_of_experience']}",
        "{input_dict['skills']}", "{input_dict['technologies']}", "{input_dict['profile']}", "{input_dict['summary']}");
    """
    with engine.connect() as connection:
        connection.execute(text(query_sql))
        connection.commit()
    # print("\nData Written to resumes_profile_data table")

def evaluate_candidates(query, language):
    result = search_resumes(query)
    responses = []  # List to store responses for each candidate
    for resume_str in result:
        name = resume_str[0]
        skills = f"{resume_str[1]}"
        skills = skills.replace("[", "")
        skills = skills.replace("]", "")
        technologies = f"{resume_str[2]}"
        technologies = technologies.replace("[", "")
        technologies = technologies.replace("]", "")
        context = f"Resume text: {resume_str[3]}"
        
        question = ""
        if language == "english":
            question = f"What percentage of the job requirements does the candidate meet for the following job description? answer in 3 lines only and be effcient while answering: {query}."
        elif language == "spanish":
            question = f"What percentage of the job requirements does the candidate meet for the following job description? answer in 3 lines only, be effcient while answering, and answer it in spanish: {query}."
        else:
            print("Language not supported!")
                       
        prompt = f"""
            Read below candidate information about the candidate:
            {context} And, the next skills: {skills}. And, next technologies: {technologies}
            Question: {question}
        """

        # print(f"prompt = {prompt}")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a expert HR analyst and recruiter.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.2,
            n=1,  # assuming you want one generation per document
        )
        # Extract the generated response
        response_text = response.choices[
            0
        ].message.content  # response['choices'][0]['message']['content']
        responses.append(
            (name, response_text)
        )  # Append the name and response_text to the responses list
    return responses

def search_resumes(query):
    query_sql = f"""
            SELECT names, skills, technologies, resume_summary FROM resumes_profile_data;
    """
    # print(query_sql, "\n")
    engine = create_engine("mysql://root:password@localhost:3306/resume_evaluator")
    connection = engine.connect()
    result = connection.execute(text(query_sql)).fetchall()
    connection.close()
    engine.dispose()
    return result

def clean_database():
    query_sql = f"""
            DELETE FROM resumes_profile_data;
    """

    engine = create_engine("mysql://root:password@localhost:3306/resume_evaluator")
    connection = engine.connect()
    connection.execute(text(query_sql))
    connection.close()
    engine.dispose()
    
def asr_request(language, audio_paths):
    if language == "english":
        path = audio_paths[0]
        language_code = "en-US"
    elif language == "spanish":
        path = audio_paths[1]
        language_code = "es-US"
    else:
        print("Language not supported!")
    
    # Create a Riva Client
    auth = riva.client.Auth(uri='localhost:50051')
    riva_asr = riva.client.ASRService(auth)
    
    # Read audio         
    with io.open(path, 'rb') as fh:
        content = fh.read()
    ipd.Audio(path)
    
    # Set up an offlinerecognition request
    config = riva.client.RecognitionConfig()
    config.language_code = language_code # Language code of the audio clip
    config.max_alternatives = 1    # How many top-N hypotheses to return
    config.enable_automatic_punctuation = True # Add punctuation when end of VAD detected
    config.audio_channel_count = 1 
    
    response = riva_asr.offline_recognize(content, config)
    asr_best_transcript = response.results[0].alternatives[0].transcript
    # print("ASR Transcript:", asr_best_transcript)
    
    return asr_best_transcript
    
def tts_request(language, text):
    timestamp = calendar.timegm(time.gmtime())
    human_readable = datetime.datetime.fromtimestamp(timestamp).isoformat()
    
    path = os.getcwd()
    path_output = path + "/audio_output/"
    
    if language == "english":
        voice_name = "English-US.Male-1"
        language_code = "en-US"
        text_input = f"This is an evaluation for {text[0]}...{text[1]}."
        audio_output = f"{path_output}english_{text[0]}_{str(human_readable)}.wav"
    elif language == "spanish":
        voice_name = "Spanish-US.Male-1"
        language_code = "es-US"
        text_input = f"Esta es una evaluacion para {text[0]}...{text[1]}."
        audio_output = f"{path_output}spanish_{text[0]}_{str(human_readable)}.wav"
    else:
        print("Language not supported!")
    
    # Create a Riva Client
    auth = riva.client.Auth(uri='localhost:50051')
    riva_tts = riva.client.SpeechSynthesisService(auth)
    
    # Setup the TTS API parameters
    sample_rate_hz = 44100
    req = { 
            "language_code"  : language_code,
            "encoding"       : riva.client.AudioEncoding.LINEAR_PCM , 
            "sample_rate_hz" : sample_rate_hz,                         
            "voice_name"     : voice_name
    }
    
    # Make a request to the Riva server
    req["text"] = text_input
    resp = riva_tts.synthesize(**req)
    audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
    wavfile.write(audio_output, sample_rate_hz, audio_samples)