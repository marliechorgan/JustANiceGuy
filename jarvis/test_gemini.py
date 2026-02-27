import asyncio
import os
from livekit.plugins import google
from livekit.agents import llm
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()

async def main():
    llm_model = google.LLM(
        model="gemini-3-flash-preview",
        api_key=os.environ["GEMINI_API_KEY"],
        thinking_config=genai_types.ThinkingConfig(thinking_level="LOW"),
    )
    
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content="You are a voice assistant named JARVIS. Your interface with users will be voice. You should use short and concise responses, and avoiding usage of unpronouncable punctuation.")
    chat_ctx.add_message(role="user", content="Hello. How are you doing? Can you check my emails?")
    
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    from agent import VoiceTools, VoiceLLMQueue
    
    queue = VoiceLLMQueue()
    tools = VoiceTools(queue)
    tools_context = llm.ToolContext(tools=llm.find_function_tools(tools))

    print("Calling chat with VoiceTools AND thinking config...")
    try:
        response = llm_model.chat(chat_ctx=chat_ctx, tools=tools_context.flatten())
        async for chunk in response:
            delta = chunk.delta
            if delta:
                if delta.content:
                    print(delta.content, end="")
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    print(f"Tool calls: {delta.tool_calls}")
    except Exception as e:
        print(f"Error Type: {type(e).__name__}")
        print(f"Error args: {e.args}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
