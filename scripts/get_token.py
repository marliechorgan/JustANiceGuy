"""Generate a LiveKit token for testing JARVIS in the playground."""
import os
from dotenv import load_dotenv
from livekit import api

load_dotenv()

api_key = os.getenv("LIVEKIT_API_KEY")
api_secret = os.getenv("LIVEKIT_API_SECRET")
url = os.getenv("LIVEKIT_URL")

if not all([api_key, api_secret, url]):
    print("Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET in .env")
    exit(1)

token = api.AccessToken(api_key, api_secret)
token.with_identity("test-user")
token.with_name("Test User")
token.with_grants(api.VideoGrants(room_join=True, room="jarvis-room"))
token.with_room_config(
    api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name="JARVIS")])
)

print(f"\nURL: {url}")
print(f"Token: {token.to_jwt()}")
print("\nTest at: https://agents-playground.livekit.io/")
