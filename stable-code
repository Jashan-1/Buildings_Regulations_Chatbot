import os
import io
import requests
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import urllib.parse
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.switch_page_button import switch_page
import json
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import time
import random
import aiohttp
import asyncio
from PyPDF2 import PdfWriter

load_dotenv()

# ---------------------- Configuration ----------------------
st.set_page_config(page_title="Building Regulations Chatbot", layout="wide", initial_sidebar_state="expanded")
# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------- Session State Initialization ----------------------

if 'pdf_contents' not in st.session_state:
    st.session_state.pdf_contents = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_pdfs' not in st.session_state:
    st.session_state.processed_pdfs = False
if 'id_counter' not in st.session_state:
    st.session_state.id_counter = 0
if 'assistant_id' not in st.session_state:
    st.session_state.assistant_id = None
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'file_ids' not in st.session_state:
    st.session_state.file_ids = []


# ---------------------- Helper Functions ----------------------

def get_vector_stores():
    try:
        vector_stores = client.beta.vector_stores.list()
        return vector_stores
    except Exception as e:
        return f"Error retrieving vector stores: {str(e)}"


def fetch_pdfs(city_code):
    url = f"http://91.203.213.50:5000/oereblex/{city_code}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print("First data:", data.get('data', [])[0] if data.get('data') else None)
        return data.get('data', [])
    else:
        st.error(f"Failed to fetch PDFs for city code {city_code}")
        return None


def download_pdf(url, doc_title):
    # Add 'https://' scheme if it's missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Sanitize doc_title to create a valid filename
        sanitized_title = ''.join(c for c in doc_title if c.isalnum() or c in (' ', '_', '-')).rstrip()
        sanitized_title = sanitized_title.replace(' ', '_')
        filename = f"{sanitized_title}.pdf"

        # Ensure filename is unique by appending the id_counter if necessary
        if os.path.exists(filename):
            filename = f"{sanitized_title}_{st.session_state.id_counter}.pdf"
            st.session_state.id_counter += 1

        # Save the PDF content to a file
        with open(filename, 'wb') as f:
            f.write(response.content)

        return filename
    except requests.RequestException as e:
        st.error(f"Failed to download PDF from {url}. Error: {str(e)}")
        return None


# Helper function to upload file to OpenAI
def upload_file_to_openai(file_path):
    try:
        file = client.files.create(
            file=open(file_path, 'rb'),
            purpose='assistants'
        )
        return file.id
    except Exception as e:
        st.error(f"Failed to upload file {file_path}. Error: {str(e)}")
        return None


def create_assistant():
    assistant = client.beta.assistants.create(
        name="Building Regulations Assistant",
        instructions="You are an expert on building regulations. Use the provided documents to answer questions accurately.",
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}]
    )
    st.session_state.assistant_id = assistant.id
    return assistant.id


def format_response(response, citations):
    """Format the response with proper markdown structure."""
    formatted_text = f"""
### Response
{response}

{"### Citations" if citations else ""}
{"".join([f"- {citation}\n" for citation in citations]) if citations else ""}
"""
    return formatted_text.strip()

def response_generator(response, citations):
    """Generator for streaming response with structured output."""
    # First yield the response header
    yield "### Response\n\n"
    time.sleep(0.1)
    
    # Yield the main response word by word
    words = response.split()
    for i, word in enumerate(words):
        yield word + " "
        # Add natural pauses at punctuation
        if word.endswith(('.', '!', '?', ':')):
            time.sleep(0.1)
        else:
            time.sleep(0.05)
    
    # If there are citations, yield them with proper formatting
    if citations:
        # Add some spacing before citations
        yield "\n\n### Citations\n\n"
        time.sleep(0.1)
        
        for citation in citations:
            yield f"- {citation}\n"
            time.sleep(0.05)

def chat_with_assistant(file_ids, user_message):
    print("----- Starting chat_with_assistant -----")
    print("Received file_ids:", file_ids)
    print("Received user_message:", user_message)

    # Create attachments for each file_id
    attachments = [{"file_id": file_id, "tools": [{"type": "file_search"}]} for file_id in file_ids]
    print("Attachments created:", attachments)

    if st.session_state.thread_id is None:
        print("No existing thread_id found. Creating a new thread.")
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                    "attachments": attachments,
                }
            ]
        )
        st.session_state.thread_id = thread.id
        print("New thread created with id:", st.session_state.thread_id)
    else:
        print(f"Existing thread_id found: {st.session_state.thread_id}. Adding message to the thread.")
        message = client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=user_message,
            attachments=attachments
        )
        print("Message added to thread with id:", message.id)

    try:
        thread = client.beta.threads.retrieve(thread_id=st.session_state.thread_id)
        print("Retrieved thread:", thread)
    except Exception as e:
        print(f"Error retrieving thread with id {st.session_state.thread_id}: {e}")
        return "An error occurred while processing your request.", []

    try:
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=st.session_state.assistant_id
        )
        print("Run created and polled:", run)
    except Exception as e:
        print("Error during run creation and polling:", e)
        return "An error occurred while processing your request.", []

    try:
        messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
        print("Retrieved messages:", messages)
    except Exception as e:
        print("Error retrieving messages:", e)
        return "An error occurred while retrieving messages.", []

    # Process the first message content
    if messages and messages[0].content:
        message_content = messages[0].content[0].text
        print("Raw message content:", message_content)

        annotations = message_content.annotations
        citations = []
        seen_citations = set()
        
        # Process annotations and citations
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
            if file_citation := getattr(annotation, "file_citation", None):
                try:
                    cited_file = client.files.retrieve(file_citation.file_id)
                    citation_entry = f"[{index}] {cited_file.filename}"
                    if citation_entry not in seen_citations:
                        citations.append(citation_entry)
                        seen_citations.add(citation_entry)
                except Exception as e:
                    print(f"Error retrieving cited file for annotation {index}: {e}")

        # Create a container for the response with proper styling
        response_container = st.container()
        with response_container:
            message_placeholder = st.empty()
            streaming_content = ""
            
            # Stream the response with structure
            for chunk in response_generator(message_content.value, citations):
                streaming_content += chunk
                # Use markdown for proper formatting during streaming
                message_placeholder.markdown(streaming_content + "▌")
            
            # Final formatted response
            final_formatted_response = format_response(message_content.value, citations)
            message_placeholder.markdown(final_formatted_response)
            
            return final_formatted_response, citations
    else:
        return "No response received from the assistant.", []


# ---------------------- Streamlit App ----------------------

# ---------------------- Custom CSS Injection ----------------------

# Inject custom CSS to style chat messages
st.markdown("""
    <style>
    /* Style for the chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    /* Style for individual chat messages */
    .chat-message {
        margin-bottom: 1.5rem;
    }

    /* Style for user messages */
    .chat-message.user > div:first-child {
        color: #1E90FF;  /* Dodger Blue for "You" */
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    /* Style for assistant messages */
    .chat-message.assistant > div:first-child {
        color: #32CD32;  /* Lime Green for "Assistant" */
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    /* Style for the message content */
    .message-content {
        padding: 1rem;
        border-radius: 0.5rem;
        line-height: 1.5;
    }

    .message-content h3 {
        color: #444;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .message-content ul {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        padding-left: 1.5rem;
    }

    .message-content li {
        margin-bottom: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

page = st.sidebar.selectbox("Choose a page", ["Documents", "Home", "Admin"])

if page == "Home":
    st.title("Building Regulations Chatbot", anchor=False)

    # Sidebar improvements
    with st.sidebar:
        colored_header("Selected Documents", description="Documents for chat")
        if 'selected_pdfs' in st.session_state and not st.session_state.selected_pdfs.empty:
            for _, pdf in st.session_state.selected_pdfs.iterrows():
                st.write(f"- {pdf['Doc Title']}")
        else:
            st.write("No documents selected. Please go to the Documents page.")

    # Main chat area improvements
    colored_header("Chat", description="Ask questions about building regulations")

    # Chat container with custom CSS class
    st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        with st.container():
            if chat['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user">
                    <div><strong>You</strong></div>
                    <div class="message-content">{chat['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div><strong>Assistant</strong></div>
                    <div class="message-content">{chat['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Inject JavaScript to auto-scroll the chat container
    st.markdown("""
        <script>
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        </script>
        """, unsafe_allow_html=True)

    # Chat input improvements
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question about building regulations...", height=100)
        col1, col2 = st.columns([3, 1])
        with col2:
            submit = st.form_submit_button("Send", use_container_width=True)

    if submit and user_input.strip() != "":
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        if not st.session_state.file_ids:
            st.error("Please process PDFs first.")
        else:
            with st.spinner("Generating response..."):
                try:
                    response, citations = chat_with_assistant(st.session_state.file_ids, user_input)
                    # The response is already formatted, so we can add it directly to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

        # Rerun the app to update the chat display
        st.rerun()

    # Footer improvements
    add_vertical_space(2)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Powered by OpenAI GPT-4 and Pinecone")
    with col2:
        st.caption("© 2023 Your Company Name")

elif page == "Documents":
    st.title("Document Selection")

    city_code_input = st.text_input("Enter city code:", key="city_code_input")
    load_documents_button = st.button("Load Documents", key="load_documents_button")

    if load_documents_button and city_code_input:
        with st.spinner("Fetching PDFs..."):
            pdfs = fetch_pdfs(city_code_input)
            if pdfs:
                st.session_state.available_pdfs = pdfs
                st.success(f"Found {len(pdfs)} PDFs")
            else:
                st.error("No PDFs found")

    if 'available_pdfs' in st.session_state:
        st.write(f"Total PDFs: {len(st.session_state.available_pdfs)}")

        # Create a DataFrame from the available PDFs
        df = pd.DataFrame(st.session_state.available_pdfs)

        # Select and rename only the specified columns
        df = df[['municipality', 'abbreviation', 'doc_title', 'file_title', 'file_href', 'enactment_date', 'prio']]
        df = df.rename(columns={
            "municipality": "Municipality",
            "abbreviation": "Abbreviation",
            "doc_title": "Doc Title",
            "file_title": "File Title",
            "file_href": "File Href",
            "enactment_date": "Enactment Date",
            "prio": "Prio"
        })

        # Add a checkbox column to the DataFrame at the beginning
        df.insert(0, "Select", False)

        # Configure grid options
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
        gb.configure_column("Select", header_name="Select", cellRenderer='checkboxRenderer')
        gb.configure_column("File Href", cellRenderer='linkRenderer')
        gb.configure_selection(selection_mode="multiple", use_checkbox=True)
        gb.configure_side_bar()
        gridOptions = gb.build()

        # Display the AgGrid
        grid_response = AgGrid(
            df,
            gridOptions=gridOptions,
            enable_enterprise_modules=True,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=False,
        )

        # Get the selected rows
        selected_rows = grid_response['selected_rows']

        # Debug: Print the structure of selected_rows
        st.write("Debug - Selected Rows Structure:", selected_rows)

        if st.button("Process Selected PDFs"):
            if len(selected_rows) > 0:  # Check if there are any selected rows
                # Convert selected_rows to a DataFrame
                st.session_state.selected_pdfs = pd.DataFrame(selected_rows)
                st.session_state.assistant_id = create_assistant()
                with st.spinner("Processing PDFs and creating/updating assistant..."):
                    file_ids = []

                    for _, pdf in st.session_state.selected_pdfs.iterrows():
                        # Debug: Print each pdf item
                        st.write("Debug - PDF item:", pdf)

                        file_href = pdf['File Href']
                        doc_title = pdf['Doc Title']

                        # Pass doc_title to download_pdf
                        file_name = download_pdf(file_href, doc_title)
                        if file_name:
                            file_path = f"./{file_name}"
                            file_id = upload_file_to_openai(file_path)
                            if file_id:
                                file_ids.append(file_id)
                            else:
                                st.warning(f"Failed to upload {doc_title}. Skipping this file.")
                        else:
                            st.warning(f"Failed to download {doc_title}. Skipping this file.")

                    st.session_state.file_ids = file_ids
                st.success("PDFs processed successfully. You can now chat on the Home page.")
            else:
                st.warning("Select at least one PDF.")
        

elif page == "Admin":
    st.title("Admin Panel")
    st.header("Vector Stores Information")

    vector_stores = get_vector_stores()
    json_vector_stores = json.dumps([vs.model_dump() for vs in vector_stores])
    st.write(json_vector_stores)

    # Add a button to go back to the main page
















