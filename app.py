import os
from typing import Any, Callable, Dict
import sqlite3
import chainlit as cl
from chainlit.config import config
from dotenv import load_dotenv
from chainlit.cli import run_chainlit
from openai import AsyncAzureOpenAI, AzureOpenAI
from assistant_event_handler import EventHandler
import pandas as pd


load_dotenv()

# tools_list = None

instructions = (
    "You are an advanced sales analysis assistant for Contoso. Your role is to be polite, professional, helpful, and friendly while assisting users with their sales data inquiries.",
    "You get all the sales data from this app using the functions provided. This data includes sales revenue categorized by region, product category, product type, and broken down by year and month.",
    "Here are some examples of the data structure:",
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
    "- Always show data in a table format, unless a chart is specifically requested.",
    "- Don't offer download links for the data.",
    # "- Don't provide code snippets or code download links, use for code interpreter only.",
    "- Use the code interpreter tool to generate charts or visualizations",
    "Remember to maintain a professional and courteous tone throughout your interactions. Avoid sharing any sensitive or confidential information.",
)


def get_table_names(conn):
    """Return a list of table names."""
    table_names = []
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for table in tables.fetchall():
        if table[0] != "sqlite_sequence":
            table_names.append(table[0])
    return table_names


def get_column_info(conn, table_name):
    """Return a list of tuples containing column names and their types."""
    column_info = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_info.append(f"{col[1]}: {col[2]}")  # col[1] is the column name, col[2] is the column type
    return column_info


def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_info(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts


@cl.step(type="tool")
def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    results = []

    try:
        data = pd.read_sql_query(query, conn)
        # data = conn.execute(query).fetchall()
        if data.empty:
            results.append(f"Query: {query}\n\n")
            results.append("No results found.")
            return results
        results.append(f"Query: {query}\n\n")
        table = data.to_string(index=False)
        # print(type(data))
        # print(data)

        results.append(table)

    except Exception as e:
        results.append(f"Query: {query}")

        results.append(f"query failed with error: {e}")
    return results


ENDPOINT_URL = os.environ.get("ENDPOINT_URL")
API_KEY = os.environ.get("API_KEY")
API_VERSION = os.environ.get("API_VERSION")
OPENAI_ASSISTANT_ID = os.environ.get("OPENAI_ASSISTANT_ID")
api_deployment_name = os.getenv("OPENAI_GPT_DEPLOYMENT")


async_openai_client = AsyncAzureOpenAI(
    azure_endpoint=ENDPOINT_URL,
    api_key=API_KEY,
    api_version=API_VERSION,
)

cl.instrument_openai()


assistant = None
sql_connection = None


def initialize(instructions: tuple, api_deployment_name: str):

    sql_connection = sqlite3.connect("./database/contoso-sales.db")

    database_schema_dict = get_database_info(sql_connection)
    database_schema_string = "\n".join(
        [f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}" for table in database_schema_dict]
    )

    tools_list = [
        {"type": "code_interpreter"},
        {
            "type": "function",
            "function": {
                "name": "ask_database",
                "description": "Use this function to answer user questions about contoso sales data. Input should be a fully formed SQL query.",
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
        azure_endpoint=ENDPOINT_URL,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    assistant = sync_openai_client.beta.assistants.create(
        name="Portfolio Management Assistant",
        model=api_deployment_name,
        instructions=str(instructions),
        tools=tools_list,
    )

    config.ui.name = assistant.name

    return assistant, sql_connection


function_map: Dict[str, Callable[[Any], str]] = {
    "ask_database": lambda args: ask_database(conn=sql_connection, query=args.get("query")),
    # "get_sales_by_month": lambda args: get_sales_by_month(args.get("region")),
}


def set_call_id(status):
    cl.user_session.set("status", status)


@cl.on_chat_start
async def start_chat():
    global assistant, sql_connection
    # lazy load the assistant
    if not assistant:
        assistant, sql_connection = initialize(instructions, api_deployment_name)
    # Create a Thread
    thread = await async_openai_client.beta.threads.create()
    # Store thread ID in user session for later use
    cl.user_session.set("thread_id", thread.id)
    # await cl.Avatar(name=assistant.name, path="./public/logo.png").send()
    await cl.Message(content=f"Hello, I'm {assistant.name}!").send()


@cl.on_chat_end
async def end_chat():
    thread_id = cl.user_session.get("thread_id")
    await async_openai_client.beta.threads.delete(thread_id=thread_id)


@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    MAX = 5
    conversation_count = 0
    last_call_id = None
    event_handler = None

    # attachments = await process_files(message.elements)

    set_call_id(None)

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
            set_call_id=set_call_id,
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


if __name__ == "__main__":

    run_chainlit(__file__)
