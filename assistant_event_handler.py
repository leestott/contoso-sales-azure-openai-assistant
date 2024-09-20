import asyncio
import json
from typing_extensions import override
from openai import AsyncAssistantEventHandler
import openai
from openai.types.beta.assistant_stream_event import ThreadMessageDelta
from openai.types.beta.threads.runs.function_tool_call import FunctionToolCall
from openai.types.beta.threads import TextDeltaBlock, ImageFileDeltaBlock
import chainlit as cl
from literalai.helper import utc_now
from sales_data import QueryResults


class EventHandler(AsyncAssistantEventHandler):
    def __init__(self, function_map: dict, assistant_name: str) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name
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
        async_openai_client = cl.user_session.get("openai-client")
        image_id = image_file.file_id
        response = await async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(name=image_id, content=response.content, display="inline", size="large")
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()

    async def process_event(self, async_openai_client, event):
        if not isinstance(event, ThreadMessageDelta):
            return

        content_block = event.data.delta.content[0]

        if isinstance(content_block, TextDeltaBlock):
            await self.current_message.stream_token(content_block.text.value)
        elif isinstance(content_block, ImageFileDeltaBlock):
            content = await async_openai_client.files.content(content_block.image_file.file_id)
            await async_openai_client.files.delete(content_block.image_file.file_id)
            image = cl.Image(content=content.content, display="inline", size="large")
            await cl.Message(content="", elements=[image]).send()
        else:
            print(type(content_block))

    @override
    async def on_tool_call_done(self, tool_call: FunctionToolCall) -> None:
        MAX_RETRIES = 5 
        async_openai_client = cl.user_session.get("openai-client")

        if tool_call.type == "function":
            tool_outputs = []
            function = self.function_map.get(tool_call.function.name)
            arguments = json.loads(tool_call.function.arguments)

            result: QueryResults = function(arguments)

            self.current_step.language = "sql"
            await self.current_step.stream_token(f"Function Name: {tool_call.function.name}\n")
            await self.current_step.stream_token(f"Function Arguments: {tool_call.function.arguments}\n\n")
            await self.current_step.stream_token(result.display_format)

            try:
                self.current_message = await cl.Message(author=self.assistant_name, content="").send()
                tool_outputs.append({"tool_call_id": tool_call.id, "output": result.json_format})

                async with async_openai_client.beta.threads.runs.submit_tool_outputs_stream(
                    thread_id=self.current_run.thread_id,
                    run_id=self.current_run.id,
                    tool_outputs=tool_outputs,
                    timeout=90,
                ) as stream:
                    async for event in stream:
                        await self.process_event(async_openai_client, event)

                    await self.current_message.update()

                for _ in range(MAX_RETRIES):
                    result = await async_openai_client.beta.threads.runs.retrieve(
                        thread_id=self.current_run.thread_id,
                        run_id=self.current_run.id,
                    )
                    if result.status == "completed":
                        break
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError("The operation took too long and was not completed within the maximum retries.")

            except openai.error.OpenAIError as e:
                print(f"Error submitting tool outputs: {e}")

        self.current_step.end = utc_now()
        await self.current_step.update()
