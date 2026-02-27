import asyncio
import os
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe, voice

load_dotenv()

async def entrypoint(ctx: JobContext):
    
    agent = voice.Agent(
        instructions="Test",
        stt=None,
        vad=None,
        llm=None,
        tts=None
    )
    
    session = voice.AgentSession()
    
    @session.on("user_speech_committed")
    def on_user_speech_committed(msg):
        print(f"User speech detected: {msg.content}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await session.start(agent, room=ctx.room)
    
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
