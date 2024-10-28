import os
import json
import spacy

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

def merge_short_sentences(text, min_words=6):
    """
    Tokenize text into sentences using spaCy and merge sentences shorter
    than the minimum word count with the following sentence.
    """
    doc = nlp(text)
    merged_sentences = []
    current_sentence = ""

    for sent in doc.sents:
        # Strip leading/trailing whitespace from the sentence text
        sentence_text = sent.text.strip()

        # If the current sentence is empty, start a new one
        if not current_sentence:
            current_sentence = sentence_text
        else:
            # If the current sentence has fewer than min_words, merge with the next
            if len(current_sentence.split()) < min_words:
                current_sentence += " " + sentence_text
            else:
                # Otherwise, add the current sentence to the list and start a new one
                merged_sentences.append(current_sentence)
                current_sentence = sentence_text

    # Append the last sentence if not empty
    if current_sentence:
        merged_sentences.append(current_sentence)

    return [s.strip() for s in merged_sentences]

def process_directory(input_dir, output_jsonl):
    """
    Process text files in a given directory, split into sentences, and save to a JSONL file.
    """
    metadata = []
    current_id = 0

    # Loop through all files in the directory
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split text into sentences, merging short ones with the next
            sentences = merge_short_sentences(text)
            
            # Create metadata for each sentence
            for sentence in sentences:
                metadata.append({
                    "ID": current_id,
                    "document": file_name,
                    "text": sentence
                })
                current_id += 1

    # Save metadata to JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry) + '\n')
    print(f'Saved processed sentences to {output_jsonl}')
    
    return metadata
