import os
from typing import Any, Callable, Dict
import chainlit as cl
from chainlit.config import config
from openai import AsyncAzureOpenAI, AzureOpenAI
from assistant_event_handler import EventHandler
from sales_data import SalesData
from dotenv import load_dotenv

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
OPENAI_ASSISTANT_ID = os.environ.get("OPENAI_ASSISTANT_ID")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")


sales_data = SalesData()


def initialize():
    database_schema_string = sales_data.get_database_info()

    instructions = (
        "You are an advanced sales analysis assistant for Contoso. Your role is to be polite, professional, helpful, and friendly while assisting users with their sales data inquiries. ",
        "You get all the sales data from the ask_database function tool in json format. ",
        f"This is the sqlite database sales data schema:{database_schema_string}. ",
        "If the user requests help or types 'help,' provide a list of sample questions that you are equipped to answer. "
        "If a question is not related to sales or is outside your scope, respond with 'I'm unable to assist with that. Please contact IT for more assistance.' ",
        "If the user is angry or insulting, remain calm and professional. Respond with, 'I'm here to help you. Let's focus on your sales data inquiries. If you need further assistance, please contact IT for support.' ",
        "You have access to a sandboxed environment for writing and testing code. "
        "If a visualization is not requested, then always show data in table format. "
        "When you are asked to create a visualization you should follow these steps: "
        "1. Write the code. "
        "2. Anytime you write new code display a preview of the code to show your work. "
        "3. Run the code to confirm that it runs. "
        "4. If the code is successful display the visualization. "
        "5. If the code is unsuccessful display the error message and try to revise the code and rerun going through the steps from above again. ",
    )

    print(str(instructions))

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
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    return sync_openai_client.beta.assistants.create(
        name="Portfolio Management Assistant",
        model=AZURE_OPENAI_DEPLOYMENT,
        instructions=str(instructions),
        tools=tools_list,
    )


assistant = initialize()

cl.instrument_openai()
config.ui.name = assistant.name


function_map: Dict[str, Callable[[Any], str]] = {
    "ask_database": lambda args: sales_data.ask_database(query=args.get("query")),
    # "get_sales_by_month": lambda args: get_sales_by_month(args.get("region")),
}


@cl.on_chat_start
async def start_chat():

    async_openai_client = AsyncAzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Create a Thread
    thread = await async_openai_client.beta.threads.create()

    # Update session state
    cl.user_session.set("openai-client", async_openai_client)
    cl.user_session.set("thread_id", thread.id)

    await cl.Message(content=f"Hello, I'm {assistant.name}!").send()


@cl.on_chat_end
async def end_chat():
    async_openai_client = cl.user_session.get("openai-client")
    thread_id = cl.user_session.get("thread_id")
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

    # Add a Message to the Thread
    await async_openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=message.content,
        # attachments=attachments,
    )

    event_handler = EventHandler(
        function_map=function_map,
        assistant_name=assistant.name,
    )

    # Create and Stream a Run
    async with async_openai_client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant.id,
        event_handler=event_handler,
        # temperature=0.4,
    ) as stream:
        await stream.until_done()
