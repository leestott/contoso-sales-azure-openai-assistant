import json
from typing_extensions import override
from openai import AsyncAssistantEventHandler
import openai
import chainlit as cl
from literalai.helper import utc_now


class EventHandler(AsyncAssistantEventHandler):
    def __init__(self, async_openai_client, assistant_name: str, function_map: dict, content: str, set_call_id) -> None:
        super().__init__()
        self.current_message: cl.Message = None
        self.current_step: cl.Step = None
        self.current_tool_call = None
        self.assistant_name = assistant_name
        self.function_map = function_map
        self.content = content
        self.set_call_id = set_call_id
        self.async_openai_client = async_openai_client
        self.tool_calls = []  # List to track tool call IDs
        self.accumulated_arguments = ""
        self.set_call_id(None)

    @override
    async def on_text_created(self, text) -> None:
        self.current_message = await cl.Message(author=self.assistant_name, content="").send()

    @override
    async def on_text_delta(self, delta, snapshot):
        if delta.value:
            await self.current_message.stream_token(delta.value)

    @override
    async def on_text_done(self, text):
        await self.current_message.update()

    @override
    async def on_tool_call_created(
        self,
        tool_call: openai.types.beta.threads.runs.function_tool_call.FunctionToolCall,
    ):
        self.current_tool_call = tool_call.id
        self.tool_calls.append(tool_call)  # Track the tool call ID
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
                        error_step = cl.Step(name=delta.type, type="tool")
                        error_step.is_error = True
                        error_step.output = output.logs
                        error_step.language = "markdown"
                        error_step.start = self.current_step.start
                        error_step.end = utc_now()
                        await error_step.send()

    @override
    async def on_tool_call_done(
        self, tool_call: openai.types.beta.threads.runs.function_tool_call.FunctionToolCall
    ) -> None:
        tool_outputs = []

        if tool_call.type == "function" and self.current_run.status == "requires_action":
            # self.set_call_id(tool_call.id, None)

            function = self.function_map.get(tool_call.function.name)
            arguments = json.loads(tool_call.function.arguments)

            result = function(arguments)

            self.current_step.language = "sql"
            await self.current_step.stream_token(f"Function Name: {tool_call.function.name}\n")
            await self.current_step.stream_token(f"Function Arguments: {tool_call.function.arguments}\n\n")
            await self.current_step.stream_token(result[0])
            await self.current_step.stream_token(result[1])

            # create the tool output to be added to the thread
            tool_outputs.append({"tool_call_id": tool_call.id, "output": result[1]})

            self.set_call_id(tool_call.id)

        for tool_call in self.tool_calls:
            if tool_call.type == "function":
                print(f"Tool Call Function: {tool_call.function.name}")
                print(f"Tool Call Arguments: {tool_call.function.arguments}")

                await self.async_openai_client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.current_run.thread_id,
                    run_id=self.current_run.id,
                    tool_outputs=tool_outputs,
                )

                # Poll for the tool submit_tool_outputs call to finish
                await self.async_openai_client.beta.threads.runs.poll(
                    run_id=self.current_run.id, thread_id=self.current_run.thread_id, timeout=60
                )

        self.current_step.end = utc_now()
        await self.current_step.update()

    async def on_image_file_done(self, image_file):
        image_id = image_file.file_id
        response = await self.async_openai_client.files.with_raw_response.content(image_id)
        image_element = cl.Image(name=image_id, content=response.content, display="inline", size="large")
        if not self.current_message.elements:
            self.current_message.elements = []
        self.current_message.elements.append(image_element)
        await self.current_message.update()
