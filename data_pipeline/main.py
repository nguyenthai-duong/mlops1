import base64
import fitz  # PyMuPDF
import re
import json
import os
from google.cloud import storage
from langchain.text_splitter import RecursiveCharacterTextSplitter

def remove_space_redundant(text):
    words = text.split()
    clean_text = " ".join(words)
    return clean_text

def get_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_chunk_json(text):
    content_between_chapters = re.findall(r"(Chương \b(?:I{1,3}(?:V?X?)?|VI{0,3}|XI{0,3}V?|XVI{0,3})\b\.?)(.*?)(?=(Chương \b(?:I{1,3}(?:V?X?)?|VI{0,3}|XI{0,3}V?|XVI{0,3})\b\.? |$))", text, re.DOTALL)
    chapter_name = []
    content_chapter = []
    all_content_chapter = []
    for content_between_chapter in content_between_chapters:
        chapter_name_temp = content_between_chapter[0].strip()
        content_chapter_temp = content_between_chapter[1].strip()
        chapter_name.append(chapter_name_temp.strip())
        content_chapter.append(content_chapter_temp.strip())
        all_content_chapter.append(content_between_chapter[0] + content_between_chapter[1])

    chapter_title = []
    rule_title = []
    contents = []
    regex_chapter = re.compile(r'(Chương \b(?:I{1,3}(?:V?X?)?|VI{0,3}|XI{0,3}V?|XVI{0,3})\b\.?)\s*(.*)')
    regex_rule = re.compile(r'(Điều \d+\.)(.*?)(?=(Điều \d+\. |$))', re.DOTALL)
    for content_chap in all_content_chapter:
        matches_chapter = regex_chapter.findall(content_chap)
        matches_rule = regex_rule.findall(content_chap)
        for match_rule in matches_rule:
            for match_chapter in matches_chapter:
                temp = match_chapter[0] + "\n" + match_chapter[1]
                chapter_title.append(temp.strip())
            temp_title_rule = match_rule[0] + match_rule[1].split('\n')[0].strip()
            rule_title.append(temp_title_rule.strip())
            temp_content_rule = remove_space_redundant(" ".join(match_rule[1].split('\n')[1:]).strip())
            contents.append(temp_content_rule)

    titles = []
    for i in range(len(chapter_title)):
        titles.append("Document Title" + "\n" + chapter_title[i] + "\n" + rule_title[i])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64)

    title_chunk, chunks = [], []
    for i in range(len(contents)):
        chunk = text_splitter.split_text(contents[i])

        num = len(chunk)
        for k in range(num):
            title_chunk.append(titles[i])
        chunks.append(chunk)

    chunks = [item for chunk in chunks for item in chunk]

    data = []
    for i in range(len(title_chunk)):
        data.append({'title': title_chunk[i], 'context': chunks[i]})

    return data

def process_pdf_file(event, context):
    """Triggered by a Pub/Sub message when a PDF file is uploaded."""
    pubsub_message = base64.b64decode(event['data']).decode('utf-8')
    message_data = json.loads(pubsub_message)

    if 'bucket' not in message_data or 'name' not in message_data:
        print("Missing 'bucket' or 'name' in Pub/Sub message")
        return

    bucket_name = message_data['bucket']
    file_name = message_data['name']

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    temp_pdf_path = f"/tmp/{file_name}"
    temp_json_path = temp_pdf_path.replace('.pdf', '.json')

    # Download the uploaded PDF file from GCS
    blob.download_to_filename(temp_pdf_path)

    # Extract text from the PDF and create a JSON
    text = get_text_from_pdf(temp_pdf_path)
    current_file_json = create_chunk_json(text)

    # Process all other PDFs in the bucket and generate a combined JSON
    all_text = ""
    blobs = bucket.list_blobs()
    for blob in blobs:
        if blob.name.endswith('.pdf') and blob.name != file_name:
            temp_pdf_path = f"/tmp/{blob.name}"
            blob.download_to_filename(temp_pdf_path)
            text = get_text_from_pdf(temp_pdf_path)
            all_text += text + "\n"

    other_files_json = create_chunk_json(all_text)

    combined_json = current_file_json + other_files_json

    if bucket.blob('all.json').exists():
        bucket.blob('all.json').delete()

    output_json_path = "/tmp/all.json"
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(combined_json, file, ensure_ascii=False, indent=4)

    output_blob = bucket.blob('all.json')
    output_blob.upload_from_filename(output_json_path)

    print(f"Processed {file_name} and combined result saved to all.json")

def handle_pdf_delete(event, context):
    """Triggered by a Pub/Sub message when a PDF file is deleted."""
    deleted_file_name = event['name']
    bucket_name = event['bucket']

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Re-generate the all.json by processing all remaining PDFs
    all_text = ""
    blobs = bucket.list_blobs()
    for blob in blobs:
        if blob.name.endswith('.pdf'):
            temp_pdf_path = f"/tmp/{blob.name}"
            blob.download_to_filename(temp_pdf_path)
            text = get_text_from_pdf(temp_pdf_path)
            all_text += text + "\n"

    updated_json = create_chunk_json(all_text)

    output_json_path = "/tmp/all.json"
    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(updated_json, file, ensure_ascii=False, indent=4)

    output_blob = bucket.blob('all.json')
    output_blob.upload_from_filename(output_json_path)

    print(f"Updated all.json after deleting {deleted_file_name}")
