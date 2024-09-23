import asyncio
import json
from typing_extensions import override
from openai import AsyncAssistantEventHandler
from openai.types.beta.threads.runs.function_tool_call import FunctionToolCall
import chainlit as cl
from literalai.helper import utc_now
from sales_data import QueryResults


class EventHandler(AsyncAssistantEventHandler):
    def __init__(self, function_map: dict, assistant_name: str, async_openai_client) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name
        self.async_openai_client = async_openai_client
        self.function_map = function_map

    @override
    async def on_text_created(self: "EventHandler", text) -> None:
        self.current_message = await cl.Message(author=self.assistant_name, content="").send()

    @override
    async def on_text_delta(self: "EventHandler", delta, snapshot):
        if delta.value:
            await self.current_message.stream_token(delta.value)

    @override
    async def on_text_done(self: "EventHandler", text: str) -> None:
        await self.current_message.update()

    @override
    async def on_tool_call_created(self: "EventHandler", tool_call: FunctionToolCall) -> None:
        self.current_tool_call = tool_call.id
        self.current_step = cl.Step(name=tool_call.type, type="tool")
        self.current_step.language = "python"
        self.current_step.created_at = utc_now()
        await self.current_step.send()

    @override
    async def on_tool_call_delta(self, delta, snapshot):
        if snapshot.id != self.current_tool_call:
            self.current_tool_call = snapshot.id
            self.current_step = cl.Step(name=delta.type, type="tool")
            self.current_step.language = "python"
            self.current_step.start = utc_now()
            await self.current_step.send()

        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                await self.current_step.stream_token(delta.code_interpreter.input)
            if delta.code_interpreter.outputs:
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        pass

    async def on_image_file_done(self, image_file):
        # async_openai_client = cl.user_session.get("openai-client")
        image_id = image_file.file_id
        response = await self.async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(name=image_id, content=response.content, display="inline", size="large")
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()

    @override
    async def on_tool_call_done(self, tool_call: FunctionToolCall) -> None:

        if tool_call.type == "function":
            try:
                function = self.function_map.get(tool_call.function.name)
                arguments = json.loads(tool_call.function.arguments)

                result: QueryResults = function(arguments)

                self.current_step.language = "sql"
                await self.current_step.stream_token(f"Function Name: {tool_call.function.name}\n")
                await self.current_step.stream_token(f"Function Arguments: {tool_call.function.arguments}\n\n")
                await self.current_step.stream_token(result.display_format)

                self.current_message = await cl.Message(author=self.assistant_name, content="").send()

                tool_outputs = []
                tool_outputs.append({"tool_call_id": tool_call.id, "output": result.json_format})

                async with self.async_openai_client.beta.threads.runs.submit_tool_outputs_stream(
                    thread_id=self.current_run.thread_id,
                    run_id=self.current_run.id,
                    tool_outputs=tool_outputs,
                    event_handler=EventHandler(
                        self.function_map,
                        self.assistant_name,
                        self.async_openai_client,
                    ),
                ) as stream:
                    await stream.until_done()

                await self.current_message.update()

            # triggered when the user stops a chat
            except asyncio.exceptions.CancelledError:
                if stream and stream.current_run and stream.current_run.status != "completed":
                    await self.async_openai_client.beta.threads.runs.cancel(
                        run_id=stream.current_run.id, thread_id=stream.current_run.thread_id
                    )
                    await cl.Message(content=f"Run cancelled. {stream.current_run.id}").send()
                    # break
            except Exception as e:
                await cl.Message(content=f"An error occurred: {e}").send()
                await cl.Message(content="Please try again in a moment.").send()
                # break

        self.current_step.end = utc_now()
        await self.current_step.update()
