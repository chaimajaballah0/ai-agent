import asyncio
import logging
import os
from email.header import decode_header
from typing import Optional
import requests as py_requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from client.persistence.init_db import init_db
from client.persistence.models.user import User

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Google authentication...")

# Check for required environment variables
creds_file_path = os.getenv("GMAIL_CREDS_FILE_PATH")
token_path = os.getenv("GMAIL_TOKEN_PATH")
jwt_key = os.getenv("JWT_KEY")

if not creds_file_path or not token_path:
    logger.error("Missing required environment variables:")
    if not creds_file_path:
        logger.error("GMAIL_CREDS_FILE_PATH is not set")
    if not token_path:
        logger.error("GMAIL_TOKEN_PATH is not set")
    raise ValueError("Required environment variables are not set")


def decode_mime_header(header: str) -> str:
    decoded_parts = decode_header(header)
    decoded_string = ""
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            decoded_string += part.decode(encoding or "utf-8")
        else:
            decoded_string += part
    return decoded_string


SCOPES = [
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]

# Singleton instance holder
_google_service_instance: Optional["GoogleServices"] = None

class GoogleServices:
    def __init__(
        self,
        creds_file_path: str,
        token_path: str,
        scopes: list[str] = SCOPES
    ):
        logger.info(f"Initializing GoogleServices with creds file: {creds_file_path}")
        self.creds_file_path = creds_file_path
        self.token_path = token_path
        self.scopes = scopes
        self.token = self._get_token()
        logger.info("Token retrieved successfully")
        self.user_id = None
        self.user_email = None

    def _get_token(self) -> Credentials:
        creds = None

        # Try loading existing token from file
        if os.path.exists(self.token_path):
            logger.info(f"Loading token from file: {self.token_path}")
            creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)

        # If token is expired and refreshable, refresh it silently
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Refreshing expired token...")
                creds.refresh(Request())
                return creds
            except Exception as e:
                logger.warning(f"Token refresh failed: {e}")
                creds = None  # fallback to re-authentication

        # If token doesn't exist or is invalid, do one-time manual auth via console
        if not creds or not creds.valid:
            logger.info("No valid token found. Starting console-based OAuth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(
                self.creds_file_path,
                self.scopes
            )
            creds = flow.run_console()

            # Save token to file
            with open(self.token_path, "w") as token_file:
                token_file.write(creds.to_json())
                logger.info(f"Token saved to: {self.token_path}")

        return creds

    # async def authenticate_and_store_user(self):
    #     claims = id_token.verify_oauth2_token(self.token.id_token, requests.Request())
    #     self.user_id = claims["sub"]
    #     self.user_email = claims["email"]

    #     logger.info(f"Authenticated Google user: {self.user_email} ({self.user_id})")
    #     await User.save_if_not_exists(user_id=self.user_id, email=self.user_email)
    #     return self.user_id

    async def authenticate_and_store_user(self):
        logger.info("Fetching user profile from Google UserInfo endpoint...")

        response = py_requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {self.token.token}"}
        )

        if response.status_code != 200:
            raise RuntimeError("Failed to fetch user info")

        profile = response.json()
        self.user_id = profile["sub"]
        self.user_email = profile["email"]

        logger.info(f"Authenticated Google user: {self.user_email} ({self.user_id})")
        await User.save_if_not_exists(user_id=self.user_id, email=self.user_email)
        return self.user_id

    def get_gmail_service(self):
        try:
            return build("gmail", "v1", credentials=self.token)
        except HttpError as error:
            logger.error(f"An error occurred building Gmail service: {error}")
            raise

def get_google_service_instance() -> GoogleServices:
    global _google_service_instance
    if _google_service_instance is None:
        logger.info("Creating GoogleServices singleton")
        _google_service_instance = GoogleServices(
            creds_file_path=creds_file_path,
            token_path=token_path,
        )
    return _google_service_instance

async def start_email_service():
    google_service = get_google_service_instance()
    await google_service.authenticate_and_store_user()
    return google_service

async def main():
    await init_db()

    google_service = get_google_service_instance()

    await google_service.authenticate_and_store_user()
    print(
        f"\n✅ User authenticated:\n - Email: {google_service.user_email}\n - User ID: {google_service.user_id}"
    )


if __name__ == "__main__":
    asyncio.run(main())
