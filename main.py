import argparse

import helpers
from helpers import (
    print_pdf_text,
    pinfo_extractor,
    add_data_to_db,
    evaluate_candidates,
    clean_database,
    asr_request,
    tts_request,
)

FLAGS = None


def main():
    # CVs to evaluate
    file_paths = [
        "/home/jangu/poc-nvidia-riva/poc-riva-code/resumes_input/Julio_Angulo_-_Research_Engineer.pdf"
    ]

    # Audios with job description
    audio_paths = [
        "/home/jangu/poc-nvidia-riva/poc-riva-code/audio_input/ml_position_english.wav",
        "/home/jangu/poc-nvidia-riva/poc-riva-code/audio_input/frontend_position_spanish.wav",
    ]

    # Resume extractor
    for file_path in file_paths:
        resume_text = print_pdf_text(file_path=file_path).replace("\n", " ")
        ip_data_dict = pinfo_extractor(resume_text)
        add_data_to_db(ip_data_dict)

    # From audio to text
    job_description = asr_request(FLAGS.language, audio_paths)

    # print(f'ASR transcript = {job_description}')

    responses = evaluate_candidates(job_description, FLAGS.language)

    for response in responses:
        tts_request(FLAGS.language, response)

    # Clean db
    clean_database()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help="""\
      Language to use.
      """,
    )
    FLAGS, unparsed = parser.parse_known_args()

    main()
