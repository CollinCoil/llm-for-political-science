from transformers import pipeline

def summarize_document_with_pipeline(file_path, model_names, max_input_length=4096, max_summary_length=200, device=0):
    """
    Summarize a document using a list of long-context T5 models with the Hugging Face pipeline.

    Parameters:
    - file_path (str): The path to the input document file.
    - model_names (list): A list of model names to use for summarization.
    - max_input_length (int): Maximum length for the input tokens (depends on the model's context size).
    - max_summary_length (int): Maximum length for the output summary.
    - device (int): Device to run the model on (-1 for CPU, 0 for the first GPU, etc.).

    Returns:
    - summaries (dict): A dictionary containing the summaries for each model.
    """
    # Prepare the input document
    with open(file_path, 'r', encoding='utf-8') as file:
        input_document = file.read()

    summaries = {}

    # Loop over each model name
    for model_name in model_names:
        print(f"Processing model: {model_name} on {'GPU' if device >= 0 else 'CPU'}")

        # Initialize the summarization pipeline with the specified device
        summarizer = pipeline("summarization", model=model_name, tokenizer=model_name, device=device)

        # Generate the summary
        summaries = summarizer(
            input_document,
            max_length=max_summary_length,
            min_length=40,
            truncation=True,
            do_sample=True,
            top_p=0.75,
            temperature=1.5,
            num_beams=3,
            num_return_sequences=3
        )

        for i, summary in enumerate(summaries):
            print(f"Summary {i+1}: {summary['summary_text']}")

    return 
