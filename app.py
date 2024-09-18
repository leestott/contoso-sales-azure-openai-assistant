import os
from typing import Any, Callable, Dict
import chainlit as cl
from chainlit.config import config
from openai import AsyncAzureOpenAI, AzureOpenAI
from assistant_event_handler import EventHandler
from sales_data import SalesData
from dotenv import load_dotenv

load_dotenv()

ENDPOINT_URL = os.environ.get("ENDPOINT_URL")
API_KEY = os.environ.get("API_KEY")
API_VERSION = os.environ.get("API_VERSION")
OPENAI_ASSISTANT_ID = os.environ.get("OPENAI_ASSISTANT_ID")
DEPLOYMENT_NAME = os.getenv("OPENAI_GPT_DEPLOYMENT")


sales_data = SalesData()




def initialize():
    database_schema_dict = sales_data.get_database_info()
    database_schema_string = "\n".join(
        [f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}" for table in database_schema_dict]
    )

    instructions = (
        "You are an advanced sales analysis assistant for Contoso. Your role is to be polite, professional, helpful, and friendly while assisting users with their sales data inquiries.",
        "You get all the sales data from this app using the functions provided. This data includes sales revenue categorized by region, product category, product type, and broken down by year and month.",
        "This is the sqlite sales data structure:{database_schema_string}.",
        # "- Regions: Africa, Asia, Europe, America",
        # "- Product Categories: Climbing gear, Camping equipment, Apparel, etc.",
        # "- Product Types: Jackets, Hammocks, Wetsuits, Crampons, Shoes, etc.",
        # "- Months: 2023-01, 2023-08, 2024-02, etc.",
        # "- Revenue: Numeric values representing the sales revenue.",
        # "- Discounts: Numeric values representing the discounts applied to the sales.",
        # "- Shipping Costs: Numeric values representing the shipping costs.",
        # "- Net Revenue: you can calculate user revenue, discount and shipping cost",
        "Your responsibilities include the following:",
        "- Analyze and provide insights based on the available sales data.",
        "- Generate visualizations that help illustrate the data trends.",
        "- If a question is not related to sales or is outside your scope, respond with 'I'm unable to assist with that. Please contact IT for more assistance.'",
        "- If the user requests help or types 'help,' provide a list of sample questions that you are equipped to answer.",
        "- If the user is angry or insulting, remain calm and professional. Respond with, 'I'm here to help you. Let's focus on your sales data inquiries. If you need further assistance, please contact IT for support.'",
        # "- Unless asked, default to formatting your responses as a table.",
        "- Don't offer download links for the data.",
        # "- Don't provide code snippets or code download links, use for code interpreter only.",
        "- Always use the code interpreter tool to generate charts or visualizations or perform complex calculations.",
        "Remember to maintain a professional and courteous tone throughout your interactions. Avoid sharing any sensitive or confidential information.",
    )

    tools_list = [
        {"type": "code_interpreter"},
        {
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "Use this function to answer user questions about contoso sales data. Input should be a fully formed SQLite3 query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": f"""
                                    SQLite3 query extracting info to answer the user's question.
                                    SQLite3 should be written using this database schema:
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
        azure_endpoint=ENDPOINT_URL,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    return sync_openai_client.beta.assistants.create(
        name="Portfolio Management Assistant",
        model=DEPLOYMENT_NAME,
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
        azure_endpoint=ENDPOINT_URL,
        api_key=API_KEY,
        api_version=API_VERSION,
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
    cl.user_session.set("openai-client", None)
    thread_id = cl.user_session.get("thread_id")
    await async_openai_client.beta.threads.delete(thread_id=thread_id)


@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    MAX = 5
    conversation_count = 0
    async_openai_client = cl.user_session.get("openai-client")

    # attachments = await process_files(message.elements)

    while conversation_count < MAX:
        conversation_count += 1

        status = cl.user_session.get("status")

        # Add a Message to the Thread
        await async_openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message.content,
            # attachments=attachments,
        )

        event_handler = EventHandler(
            async_openai_client=async_openai_client,
            assistant_name=assistant.name,
            function_map=function_map,
            content=message.content,
        )

        # Create and Stream a Run
        async with async_openai_client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=assistant.id,
            event_handler=event_handler,
        ) as stream:
            await stream.until_done()

        status = cl.user_session.get("status")

        if status is None:
            break
