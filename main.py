import gradio as gr
from langchain_community.llms import Ollama

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    # Handle file uploads
    for file_path in message["files"]:
        # Process files as needed (e.g., extract text from images, PDFs, etc.)
        history.append(((file_path,), None))
    # Handle text input
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

# Initialize the Ollama LLM
llm = Ollama(model="phi3:latest")

def bot(history):
    # Construct the conversation history as a string
    conversation = ''
    for h in history[:-1]:
        user_msg = h[0]
        bot_msg = h[1]
        if isinstance(user_msg, tuple):
            user_msg = "User uploaded a file."
        conversation += f"User: {user_msg}\n"
        if bot_msg:
            conversation += f"Assistant: {bot_msg}\n"
    # Get the last user message
    last_user_message = history[-1][0]
    if isinstance(last_user_message, tuple):
        last_user_message = "User uploaded a file."
    conversation += f"User: {last_user_message}\nAssistant:"

    # Initialize the response
    llm_response = ""
    # Stream the response from the LLM
    for chunk in llm.stream(conversation):
        llm_response += chunk
        # Update the last entry in history with the current response
        history[-1][1] = llm_response
        # Yield the updated history
        yield history

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        scale=1,
    )

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Enter message or upload file...",
        show_label=False
    )

    chat_msg = chat_input.submit(
        add_message,
        [chatbot, chat_input],
        [chatbot, chat_input]
    )
    bot_msg = chat_msg.then(
        bot,
        chatbot,
        chatbot,
        api_name="bot_response"
    )
    bot_msg.then(
        lambda: gr.MultimodalTextbox(interactive=True),
        None,
        [chat_input]
    )

    chatbot.like(print_like_dislike, None, None)

demo.launch()
