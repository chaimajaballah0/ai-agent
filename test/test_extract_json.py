import pytest
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from client.helper import extract_json


def test_extract_json_plain_object():
    text = '{"tool": "send_email", "arguments": {"recipient_id": "a@b.com"}}'
    result = extract_json(text)
    assert result is not None
    assert json.loads(result)["tool"] == "send_email"

def test_extract_json_fenced_block():
    text = """```json
    {
        "tool": "send_email",
        "arguments": {
            "recipient_id": "a@b.com"
        }
    }
    ```"""
    result = extract_json(text)
    assert result is not None
    data = json.loads(result)
    assert data["tool"] == "send_email"

def test_extract_json_embedded_in_text():
    text = """Sure! I’ll now use the email tool.
    
    ```json
    {
        "tool": "send_email",
        "arguments": {
            "recipient_id": "a@b.com",
            "subject": "Test",
            "message": "Hi there!"
        }
    }
    ```
    Let me know if that worked.
    """
    result = extract_json(text)
    assert result is not None
    data = json.loads(result)
    assert data["arguments"]["subject"] == "Test"

def test_extract_json_no_json():
    text = "Hello there! No tools needed."
    result = extract_json(text)
    assert result is None

def test_extract_json_malformed_json():
    text = """```json
    { "tool": "send_email", "arguments": { "recipient_id": "a@b.com"  }
    ```"""
    result = extract_json(text)
    assert result is None

def test_extract_json_array():
    text = """```json
    [
        {
            "tool": "send_email",
            "arguments": { "recipient_id": "a@b.com" }
        },
        {
            "tool": "trash_email",
            "arguments": { "email_id": "123" }
        }
    ]
    ```"""
    result = extract_json(text)
    print(result)
    assert result is not None
    data = json.loads(result)
    assert isinstance(data, list)
    assert data[0]["tool"] == "send_email"
