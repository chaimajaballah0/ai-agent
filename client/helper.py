import re
import json

def extract_json(text: str) -> dict | None:
    """
    Extracts JSON content enclosed in triple backticks with `json` specifier from a string.

    Args:
        text (str): The input string containing JSON.

    Returns:
        dict: The extracted JSON object.

    Raises:
        ValueError: If no JSON block is found or if parsing fails.
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in the input text.")
    
    json_str = match.group(1)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")