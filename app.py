import json
import os
from typing import Any, Callable, Dict

import chainlit as cl
from chainlit.config import config
from openai import AsyncAzureOpenAI, AzureOpenAI
from dotenv import load_dotenv
import httpx

from assistant_event_handler import EventHandler
from sales_data import SalesData

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
OPENAI_ASSISTANT_ID = os.environ.get("OPENAI_ASSISTANT_ID")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

assistant = None


def initialize(sales_data: SalesData, api_key: str):
    database_schema_string = sales_data.get_database_info()

    instructions = (
        "You are an advanced sales analysis assistant for Contoso. Your role is to be polite, professional, helpful, and friendly while assisting users with their sales data inquiries. ",
        "You get all the sales data from the ask_database function tool in json format. ",
        f"This is the sqlite database sales data schema:{database_schema_string}. ",
        "If the user requests help or types 'help,' provide a list of sample questions that you are equipped to answer. "
        "If a question is not related to sales or is outside your scope, respond with 'I'm unable to assist with that. Please contact IT for more assistance.' ",
        "If the user is angry or insulting, remain calm and professional. Respond with, 'I'm here to help you. Let's focus on your sales data inquiries. If you need further assistance, please contact IT for support.' ",
        "Don't show any code or snippets to the user. ",
        "You have access to a sandboxed environment for writing and testing code. "
        "If a visualization is not requested, then always show data in table format. "
        "When you are asked to create a visualization you should follow these steps: "
        "1. Write the code and with the users language for visualizations. "
        "2. Anytime you write new code display a preview of the code to show your work. "
        "3. Run the code to confirm that it runs. "
        "4. If the code is successful display the visualization. "
        "5. If the code is unsuccessful display the error message and try to revise the code and rerun going through the steps from above again. ",
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

    assistant = sync_openai_client.beta.assistants.create(
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
            label="Chart sales anomalies for ski gear in 2023 in Europe",
            message="Chart sales anomalies for ski gear in 2023 in Europe",
            icon="/public/write.svg",
        ),
    ]


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
        user = cl.User(identifier=username, metadata=event_settings)
        return user
    return None


sales_data = SalesData()

cl.instrument_openai()


function_map: Dict[str, Callable[[Any], str]] = {
    "ask_database": lambda args: sales_data.ask_database(query=args.get("query")),
}


@cl.on_chat_start
async def start_chat():
    global assistant
    try:
        metadata = cl.user_session.get("user").metadata
        api_key = metadata.get("api_key")

        if assistant is None:
            assistant = initialize(sales_data=sales_data, api_key=api_key)

        async_openai_client = AsyncAzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=api_key,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        thread = await async_openai_client.beta.threads.create()

        # Update session state
        cl.user_session.set("openai-client", async_openai_client)
        cl.user_session.set("thread_id", thread.id)
    except Exception as e:
        cl.user_session.set("openai-client", None)
        cl.user_session.set("thread_id", None)
        await cl.Message(content=e.response.reason_phrase).send()
        return

    # await cl.Message(content=f"Hello, I'm {assistant.name}!").send()


@cl.on_chat_end
async def end_chat():
    async_openai_client = cl.user_session.get("openai-client")
    thread_id = cl.user_session.get("thread_id")
    if thread_id:
        try:
            await async_openai_client.beta.threads.delete(thread_id=thread_id)
        except Exception as e:
            print(f"Error deleting thread: {e}")
        finally:
            cl.user_session.set("openai-client", None)


@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    async_openai_client = cl.user_session.get("openai-client")

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
            ),
            parallel_tool_calls=False,  # Disable parallel tool calls
            temperature=0.4,
        ) as stream:
            await stream.until_done()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()
