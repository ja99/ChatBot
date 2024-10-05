import gradio as gr
from langchain_community.llms import Ollama

# Additional imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = None  # Initialize vectorstore as None

# Initialize the Ollama LLM
llm = Ollama(model="phi3:latest")

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def add_message(history, message):
    global vectorstore  # Declare vectorstore as global
    # Handle file uploads
    for file_path in message["files"]:
        # Process PDF files
        if file_path.endswith(".pdf"):
            # Load the PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            # Add documents to vector store
            if vectorstore is None:
                vectorstore = FAISS.from_documents(documents, embeddings)
            else:
                vectorstore.add_documents(documents)
            history.append((f"Uploaded and processed PDF: {file_path}", None))
        else:
            # For other file types
            history.append(((file_path,), None))
    # Handle text input
    if message["text"] is not None:
        history.append((message["text"], None))
    return history, gr.MultimodalTextbox(value=None, interactive=False)

def bot(history):
    global vectorstore  # Declare vectorstore as global
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

    # Retrieve relevant documents from vectorstore
    context = ''
    if vectorstore is not None:
        relevant_docs = vectorstore.similarity_search(last_user_message, k=3)  # adjust k as needed
        # Extract the text from these documents
        context = "\n".join([doc.page_content for doc in relevant_docs])

    # Combine context and conversation
    prompt = f"{conversation}\nRelevant context:\n{context}\nAssistant:"

    # Initialize the response
    llm_response = ""
    # Stream the response from the LLM
    for chunk in llm.stream(prompt):
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