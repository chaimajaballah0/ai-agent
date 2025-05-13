import asyncio
import logging
import uuid
from authentication.auth import start_email_service

from client.assistant.planning import Planner
from client.configuration.configuration import Configuration
from client.persistence.init_db import init_db
from client.persistence.models.thread import UserThread

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


async def main() -> None:
    """Initialize and run the chat session."""
    mcp_config_path = "src/client/configuration/servers_config.json"
    config = Configuration(mcp_config_path)
    await init_db()
    google_service = await start_email_service()
    print(
        f"\n✅ User authenticated:\n - Email: {google_service.user_email}\n - User ID: {google_service.user_id}"
    )

    # Step 2: List existing sessions
    threads = await UserThread.list_threads(google_service.user_id)
    if threads:
        print("\nAvailable threads:")
        for s in threads:
            print(f"- {s.thread_id}")
    else:
        print("\nNo previous sessions found.")

    # Step 3: Let user choose or create
    choice = input("\nEnter session ID to resume or type 'new' to create one: ").strip()
    if choice.lower() == "new":
        thread_id = str(uuid.uuid4())
        planner = Planner(config, google_service.user_id, thread_id)
        await planner.setup()
    else:
        thread_id = choice
        try:
            planner = await Planner.load_session(
                config, google_service.user_id, thread_id
            )
        except Exception as e:
            logging.error(f"Failed to load session: {e}")
            return
    await planner.start()


if __name__ == "__main__":
    asyncio.run(main())
