import os
import json
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Set up logging configuration to save logs to a text file
log_filename = f"rec_sys_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),  # Save logs to a file
        logging.StreamHandler()            # Also print logs to the console
    ]
)

try:
    # Set your Gemini Pro API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        logging.info("API Key successfully set.")
    else:
        logging.error("Failed to set API Key. Please check the .env file.")
    
    # Configure the generative AI model
    genai.configure(api_key=google_api_key)
    logging.info("Model configuration successful.")
except Exception as e:
    logging.error(f"Error setting up API or model: {e}")
    raise

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

try:
    # Initialize the model once to avoid repeated setup
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-exp-0801",
        generation_config=generation_config,
    )
    logging.info("Model initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing the model: {e}")
    raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def identify_topics(paper, model=model):
    prompt = f"""
    You will be given a set of abstracts from scientific papers. Your task is to decompose each abstract into 3 unique tags that capture the main themes or topics of the research. Then, you will create a JSON object containing the title of each paper and its corresponding tags.
    Here are the abstracts:
    <abstracts>
    {paper}
    </abstracts>
    Follow these steps for each abstract:
    1. Read all the abstracts carefully.
    2. Identify the main themes, topics, or key concepts discussed in the abstract.
    3. Create 5 unique tags that best represent these themes. The tags should be concise (1-2 words) and relevant to the research described in the abstract.
    4. Extract the title of the paper from the abstract.
    After processing all abstracts, create a JSON object with the following structure:
    {{
      "papers": [
        {{
          "title": "Paper Title 1",
          "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
        }},
        {{
          "title": "Paper Title 2",
          "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
        }},
        ...
      ]
    }}
    Guidelines for creating tags:
    - Use lowercase letters for all tags
    - Avoid overly general terms (e.g., "science", "AI" " "research", "LLM", "Large Language Models",  "Generative AI",)
    - Focus on specific concepts, methods, or subjects discussed in the abstract
    - Ensure that the tags are unique within each paper's set of tags
    Your final output should be the JSON object containing only the titles and tags for each paper. Do not include any additional explanations or text outside of the JSON structure.
    """

    try:
        response = model.generate_content(prompt)
        logging.info("Successfully generated content from model.")

        # Extract the text content from the GenerateContentResponse object
        response_text = response.text
        
        logging.info(f"Raw model response: {response_text}")

        # Parse the JSON from the text content
        parsed_response = json.loads(response_text)
        
        # Validate the structure
        if not isinstance(parsed_response, dict) or "papers" not in parsed_response:
            raise ValueError("Invalid response structure")
        
        for paper in parsed_response["papers"]:
            if not all(key in paper for key in ["title", "tags"]):
                raise ValueError("Invalid paper structure")
            if len(paper["tags"]) != 5:
                raise ValueError("Each paper must have exactly 5 tags")

        return parsed_response["papers"]

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from model response: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during topic identification: {e}")
        raise

def save_topics_to_file(response, filename):
    try:
        with open(filename, "w") as f:
            json.dump(response, f, indent=2)
        logging.info(f"Tags successfully written to file: {filename}")
    except IOError as e:
        logging.error(f"Failed to save tags to file: {e}")
        raise

def analyze_preferences(papers_read):
    all_papers_data = []
    for paper in papers_read:
        try:
            topics_response = identify_topics(paper)
            all_papers_data.extend(topics_response)  # Accumulate results
        except Exception as e:
            logging.error(f"Error processing paper: {e}")
            continue

    # Save all papers data to a single JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_tags_{timestamp}.json"
    save_topics_to_file({"papers": all_papers_data}, filename)
    
    user_profile = {}
    for paper_data in all_papers_data:
        for tag in paper_data["tags"]:
            user_profile[tag] = user_profile.get(tag, 0) + 1
    
    return user_profile

def create_user_profile():
    json_path = os.path.join("great_filter", "prev_papers.json")
    try:
        logging.info(f"Attempting to read file: {json_path}")
        with open(json_path, "r") as f:
            papers_read = json.load(f)
        logging.info(f"Successfully loaded {len(papers_read)} previously read papers.")
    except FileNotFoundError:
        logging.error(f"File not found: {json_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {json_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error reading {json_path}: {e}")
        raise

    return analyze_preferences(papers_read)

if __name__ == "__main__":
    try:
        user_profile = create_user_profile()
        print("User Profile:", user_profile)
        logging.info("User profile created successfully.")
    except Exception as e:
        logging.error(f"An error occurred during execution: {e}")
        print(f"An error occurred during execution: {e}")