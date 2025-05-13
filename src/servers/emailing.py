import asyncio
import logging
import base64
from email.header import decode_header
from base64 import urlsafe_b64decode
from email import message_from_bytes
import webbrowser
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from authentication.auth import get_google_service_instance


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global placeholders
google_service = None
gmail_api = None

def decode_mime_header(header: str) -> str:
    decoded_parts = decode_header(header)
    decoded_string = ''
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            decoded_string += part.decode(encoding or 'utf-8')
        else:
            decoded_string += part
    return decoded_string


mcp = FastMCP("Email")
logger.info("MCP server created")

async def authenticate_gmail():
    global google_service, gmail_api
    google_service = get_google_service_instance()
    await google_service.authenticate_and_store_user()
    gmail_api = google_service.get_gmail_service()
    logger.info(f"Authenticated Gmail user: {google_service.user_email}")


@mcp.tool()
async def test_tool() -> dict:
    logger.info("Test tool called")
    return {"status": "success", "message": "Test tool is working"}


@mcp.tool()
async def send_email(recipient_id: str, subject: str, message: str) -> dict:
    logger.info("=== send_email tool called ===")
    try:
        message_obj = MIMEMultipart()
        message_obj.attach(MIMEText(message, 'plain'))
        message_obj['To'] = recipient_id
        message_obj['From'] = google_service.user_email
        message_obj['Subject'] = subject

        encoded_message = base64.urlsafe_b64encode(message_obj.as_bytes()).decode()
        create_message = {'raw': encoded_message}

        send_message = await asyncio.to_thread(
            gmail_api.users().messages().send(userId="me", body=create_message).execute
        )
        logger.info(f"Message sent with ID: {send_message['id']}")
        return {"status": "success", "message_id": send_message["id"]}
    except HttpError as error:
        logger.error(f"Gmail API error: {error}")
        return {"status": "error", "error_message": str(error)}
    except Exception as e:
        logger.exception("Unexpected error sending email")
        return {"status": "error", "error_message": str(e)}


@mcp.tool()
async def get_unread_emails() -> list[dict[str, str]] | str:
    try:
        user_id = 'me'
        query = 'in:inbox is:unread category:primary'

        response = gmail_api.users().messages().list(userId=user_id, q=query).execute()
        messages = response.get('messages', [])

        while 'nextPageToken' in response:
            response = gmail_api.users().messages().list(
                userId=user_id, q=query, pageToken=response['nextPageToken']
            ).execute()
            messages.extend(response.get('messages', []))

        return messages
    except HttpError as error:
        return f"An HttpError occurred: {str(error)}"


@mcp.tool()
async def read_email(email_id: str) -> dict[str, str] | str:
    try:
        msg = gmail_api.users().messages().get(userId="me", id=email_id, format='raw').execute()
        raw_data = msg['raw']
        decoded_data = urlsafe_b64decode(raw_data)
        mime_message = message_from_bytes(decoded_data)

        body = None
        if mime_message.is_multipart():
            for part in mime_message.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode()
                    break
        else:
            body = mime_message.get_payload(decode=True).decode()

        email_metadata = {
            "content": body,
            "subject": decode_mime_header(mime_message.get("subject", "")),
            "from": mime_message.get("from", ""),
            "to": mime_message.get("to", ""),
            "date": mime_message.get("date", "")
        }

        await mark_email_as_read(email_id)
        return email_metadata

    except HttpError as error:
        return f"An HttpError occurred: {str(error)}"


@mcp.tool()
async def mark_email_as_read(email_id: str) -> str:
    try:
        gmail_api.users().messages().modify(
            userId="me",
            id=email_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        return "Email marked as read."
    except HttpError as error:
        return f"An HttpError occurred: {str(error)}"


@mcp.tool()
async def trash_email(email_id: str) -> str:
    try:
        gmail_api.users().messages().trash(userId="me", id=email_id).execute()
        return "Email moved to trash successfully."
    except HttpError as error:
        return f"An HttpError occurred: {str(error)}"


@mcp.tool()
async def open_email(email_id: str) -> str:
    try:
        url = f"https://mail.google.com/#all/{email_id}"
        webbrowser.open(url, new=0, autoraise=True)
        return "Email opened in browser successfully."
    except HttpError as error:
        return f"An HttpError occurred: {str(error)}"


logger.info("All tools registered")

async def main():
    await authenticate_gmail()
    # Rest of your code here

if __name__ == "__main__":
    logger.info("Starting MCP server...")
    mcp.run(transport="stdio")
    asyncio.run(main())
    