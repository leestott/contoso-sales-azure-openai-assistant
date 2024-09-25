import asyncio
import json
import os
from typing import Any, Callable, Dict

import chainlit as cl
from chainlit.config import config
from chainlit.types import ThreadDict
from openai import AsyncAzureOpenAI, AzureOpenAI
from dotenv import load_dotenv
import httpx

from event_handler import EventHandler
from sales_data import SalesData

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
OPENAI_ASSISTANT_ID = os.environ.get("OPENAI_ASSISTANT_ID")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_ASSISTANT_ID = os.getenv("AZURE_OPENAI_ASSISTANT_ID")

assistant = None
sales_data = SalesData()
cl.instrument_openai()

function_map: Dict[str, Callable[[Any], str]] = {
    "ask_database": lambda args: sales_data.ask_database(query=args.get("query")),
}


def get_openai_client():
    metadata = cl.user_session.get("user").metadata
    api_key = metadata.get("api_key")

    return AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=api_key,
        api_version=AZURE_OPENAI_API_VERSION,
    )


async def authenticate_api_key(api_key: str):
    url = f"{AZURE_OPENAI_ENDPOINT}/eventinfo"
    headers = {"api-key": api_key}
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers)
    if response.status_code == 200:
        return response.text
    return None


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    event_response = await authenticate_api_key(password)
    if event_response:
        event_settings = json.loads(event_response)
        event_settings.update({"api_key": password})
        return cl.User(identifier=username, metadata=event_settings)
    return None


def initialize(sales_data: SalesData, api_key: str):
    database_schema_string = sales_data.get_database_info()

    instructions = (
        "You are a Contoso sales analysis assistant. Assist users with sales data inquiries politely, professionally, and with brief explanations.",
        "Access sales data using the `ask_database` function, which returns results in JSON format.",
        f"The sales database uses this SQLite schema: {database_schema_string}.",
        "If a user asks for 'help,' provide example questions related to sales data inquiries you can assist with.",
        "If a query falls outside of sales or your scope, respond with: 'I'm unable to assist with that. Please contact IT for further help.'",
        "Stay calm and professional when encountering aggressive behavior. Respond with: 'I'm here to help with sales data inquiries. For other issues, please contact IT.'",
        "You have access to a sandboxed environment for writing and testing code.",
        "Present data in markdown tables unless the user explicitly requests visualizations.",
        "Ensure all responses and visualizations match the language of the user's question.",
        "For download requests, respond with: 'The download link is provided below.'",
        "For visualizations, follow these steps:",
        "1. Write the required code.",
        "2. Run the code to ensure it works.",
        "3. Display the visualization if successful.",
        "4. If an error occurs, display it, revise the code, and try again."
    )

    tools_list = [
        {"type": "code_interpreter"},
        {
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "Use this function to answer user questions about contoso sales data. Input should be a fully formed SQLite query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                    SQLite query extracting info to answer the user's question.
                                    SQLite should be written using this database schema:
                                    {database_schema_string}
                                    The query should be returned in plain text, not in JSON.
                                    """,
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    sync_openai_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=api_key,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    assistant = sync_openai_client.beta.assistants.retrieve(assistant_id=AZURE_OPENAI_ASSISTANT_ID)

    sync_openai_client.beta.assistants.update(
        assistant_id=assistant.id,
        name="Portfolio Management Assistant",
        model=AZURE_OPENAI_DEPLOYMENT,
        instructions=str(instructions),
        tools=tools_list,
    )

    config.ui.name = assistant.name
    return assistant


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Help",
            message="help.",
            icon="/public/idea.svg",
        ),
        cl.Starter(
            label="Can you provide a monthly revenue breakdown for winter sports products in 2023 as a chart. Use vivid colors for the chart.",
            message="Can you provide a monthly revenue breakdown for winter sports products in 2023 as a chart. Use vivid colors for the chart.",
            icon="/public/learn.svg",
        ),
        cl.Starter(
            label="Kunt u een uitsplitsing geven van de maandelijkse inkomsten voor wintersportproducten in 2023? Weergeven als staafdiagram met levendige kleuren.",
            message="Kunt u een uitsplitsing geven van de maandelijkse inkomsten voor wintersportproducten in 2023? Weergeven als staafdiagram met levendige kleuren.",
            icon="/public/terminal.svg",
        ),
        cl.Starter(
            label="2023 年におけるウィンタースポーツ製品の月次売上高内訳を教えてください。鮮やかな色を使った棒グラフで表示してください。",
            message="2023 年におけるウィンタースポーツ製品の月次売上高内訳を教えてください。鮮やかな色を使った棒グラフで表示してください。",
            icon="/public/write.svg",
        ),
    ]


@cl.on_chat_start
async def start_chat():
    global assistant
    try:
        metadata = cl.user_session.get("user").metadata
        api_key = metadata.get("api_key")

        if assistant is None:
            assistant = initialize(sales_data=sales_data, api_key=api_key)

        async_openai_client = get_openai_client()
        thread_id = cl.user_session.get("thread_id")
        if not thread_id:
            thread = await async_openai_client.beta.threads.create()
            cl.user_session.set("thread_id", thread.id)

    except Exception as e:
        cl.user_session.set("thread_id", None)
        await cl.Message(content=e.response.reason_phrase).send()
        return


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    await start_chat()


@cl.on_message
async def main(message: cl.Message) -> None:
    thread_id = cl.user_session.get("thread_id")
    async_openai_client = get_openai_client()

    if not thread_id or not async_openai_client:
        await cl.Message(content="An error occurred. Please try again later.").send()
        return

    try:
        # Add a Message to the Thread
        await async_openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message.content,
        )

        # Create and Stream a Run
        async with async_openai_client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant.id,
            event_handler=EventHandler(
                function_map=function_map,
                assistant_name=assistant.name,
                async_openai_client=async_openai_client,
            ),
            # temperature=0.4,
        ) as stream:
            await stream.until_done()

    # triggered when the user stops a chat
    except asyncio.exceptions.CancelledError:
        if stream and stream.current_run and stream.current_run.status != "completed":
            await async_openai_client.beta.threads.runs.cancel(
                run_id=stream.current_run.id, thread_id=stream.current_run.thread_id
            )
            await cl.Message(content=f"Run cancelled. {stream.current_run.id}").send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()
        await cl.Message(content="Please try again in a moment.").send()
