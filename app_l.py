import json
import os
import requests
from pypdf import PdfReader # Corrected import based on common usage
import streamlit as st
from dotenv import load_dotenv
import traceback # For detailed error logging

# Import for Hugging Face
from huggingface_hub import InferenceClient

# Load environment variables from .env file
load_dotenv()

# Load Hugging Face Token

if not HUGGING_FACE_HUB_TOKEN:
    st.error("HUGGING_FACE_HUB_TOKEN not found in environment variables. Please set it in your .env file.")
    st.stop()

# Load Pushover credentials (optional)
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")

def push(text):
    """Sends a Pushover notification if credentials are available."""
    if PUSHOVER_TOKEN and PUSHOVER_USER:
        try:
            requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": PUSHOVER_TOKEN,
                    "user": PUSHOVER_USER,
                    "message": text,
                }
            )
        except Exception as e:
            st.error(f"Error sending Pushover notification: {e}")
    else:
        print("Pushover credentials not found. Message not sent:", text)
        st.warning("Pushover credentials not found. Message not sent. Set PUSHOVER_TOKEN and PUSHOVER_USER to enable notifications.")

def record_user_details(email, name="Name not provided", notes="not provided"):
    """Placeholder for recording user details."""
    message = f"Recording user: {name} with email {email} and notes: {notes}"
    push(message)
    print(message) # Also print to console for non-Pushover users
    return {"recorded": "ok", "status": f"Details for {email} noted."}

def record_unknown_question(question):
    """Placeholder for recording unknown questions."""
    message = f"Recording unknown question: {question}"
    push(message)
    print(message) # Also print to console
    return {"recorded": "ok", "status": f"Question '{question}' recorded."}

class Me:
    def __init__(self):
        try:
            self.client = InferenceClient(token=HUGGING_FACE_HUB_TOKEN)
            self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
            # It's good practice to confirm the model is usable, but a direct call here might be slow due to cold starts.
            # For now, we assume it will work if the token and model access are correct.
            #st.info(f"Using Hugging Face model: {self.model_id}")
        except Exception as e:
            st.error(f"Fatal Error: Failed to initialize Hugging Face InferenceClient: {e}")
            traceback.print_exc()
            st.stop()

        self.name = "Ganesh Neelakanta" # Replace with the actual name if different

        try:
            # Ensure the 'me' directory and its files are in the same directory as your script,
            # or provide an absolute path.
            script_dir = os.path.dirname(__file__) # Gets the directory where the script is located
            linkedin_pdf_path = os.path.join(script_dir, "me", "Ganesh_Neelakanta_og.pdf")
            summary_txt_path = os.path.join(script_dir, "me", "summary.txt")

            reader = PdfReader(linkedin_pdf_path)
            self.linkedin = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    self.linkedin += text
        except FileNotFoundError:
            self.linkedin = "LinkedIn profile information is currently unavailable."
            st.warning(f"Warning: linkedin.pdf not found at {linkedin_pdf_path}. LinkedIn data will be missing.")
        except Exception as e:
            self.linkedin = "Error reading LinkedIn profile."
            st.error(f"Error processing linkedin.pdf: {e}")

        try:
            with open(summary_txt_path, "r", encoding="utf-8") as f:
                self.summary = f.read()
        except FileNotFoundError:
            self.summary = "Summary information is currently unavailable."
            st.warning(f"Warning: summary.txt not found at {summary_txt_path}. Summary data will be missing.")
        except Exception as e:
            self.summary = "Error reading summary information."
            st.error(f"Error processing summary.txt: {e}")

    def system_prompt(self):
        """Generates the system prompt for the LLM."""
        prompt_text = f"You are a helpful and professional AI assistant representing {self.name}. "
        prompt_text += f"Your goal is to answer questions about {self.name}'s career, background, skills, and experience, based on the provided summary and LinkedIn profile. "
        prompt_text += "Be engaging and aim to provide informative answers. "
        prompt_text += "If you cannot answer a question based on the provided context, clearly state that you don't have the specific information. Do not invent answers. "
        # Tool use instructions are removed for now, as it requires custom implementation.
        # We'll re-introduce simplified tool prompting if needed later.
        prompt_text += f"\n\n## Summary for {self.name}:\n{self.summary}\n\n## LinkedIn Profile for {self.name}:\n{self.linkedin}\n\n"
        prompt_text += f"Based on this information, please chat with the user, always staying in character as an assistant for {self.name}."
        return prompt_text

    def handle_tool_call(self, function_name, function_args_dict):
        """
        Handles a simulated tool call.
        IMPORTANT: This is a placeholder. Llama 3.1 via InferenceClient won't
        produce structured tool calls like Gemini by default. You'd need to:
        1. Prompt Llama to output JSON for tool calls.
        2. Parse that JSON here.
        3. Call the actual Python functions.
        4. Format the Python function's output back into a message for Llama.
        """
        st.info(f"Attempting to handle (simulated) tool call: {function_name} with args: {function_args_dict}")
        if function_name == "record_user_details":
            return record_user_details(**function_args_dict)
        elif function_name == "record_unknown_question":
            return record_unknown_question(**function_args_dict)
        else:
            return {"error": f"Unknown tool: {function_name}", "status": "Tool not found."}

    def chat(self, message, history):
        """Handles a chat interaction with the Hugging Face model."""
        hf_messages = []

        # System prompt
        hf_messages.append({"role": "system", "content": self.system_prompt()})

        # Add existing history
        for chat_item in history:
            role = chat_item["role"] if chat_item["role"] in ["user", "assistant"] else "user"
            hf_messages.append({"role": role, "content": chat_item["content"]})

        # Add current user message
        hf_messages.append({"role": "user", "content": message})

        try:
            # st.text("Sending request to Llama 3.1 model...") # For user feedback
            response = self.client.chat_completion(
                model=self.model_id,
                messages=hf_messages,
                max_tokens=1024,  # Max new tokens to generate
                temperature=0.7, # Optional: for creativity
                # stream=False, # Keep stream=False for now for easier debugging
            )
            
            assistant_response_text = ""
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                assistant_response_text = response.choices[0].message.content.strip()
            else:
                assistant_response_text = "I received a response, but it was empty."
                st.warning("Received an empty response from the model.")

            # --- Tool Calling Section (Placeholder - Needs Major Rework for Llama 3.1) ---
            # Llama 3.1 (especially via basic Inference API) doesn't do structured tool calls like Gemini.
            # You'd need to prompt it to output JSON representing a tool call, then parse it.
            # Example: if assistant_response_text contains something like '{"tool_call": {"name": "record_user_details", "arguments": {"email": "..."}}}'
            # For now, this part is illustrative and won't trigger as-is.

            # Let's assume for this example, we will NOT attempt complex tool call parsing from Llama's text.
            # If you want Llama to use tools, you'd typically:
            # 1. Include tool descriptions in the system prompt, instructing Llama to output a specific JSON format if it wants to use a tool.
            # 2. Check the `assistant_response_text` for this JSON.
            # 3. If JSON for a tool call is found:
            #    - Extract tool name and arguments.
            #    - Call `self.handle_tool_call(tool_name, tool_args)`.
            #    - Construct a new message with the tool's response.
            #    - Make a second call to `self.client.chat_completion` with the original history, Llama's first response (the tool call JSON), and the tool execution result.
            #    - The final text from this second call would be the user-facing answer.
            # This is a multi-step process. The current code only does one call.

            return assistant_response_text

        except Exception as e:
            st.error(f"Error during chat with Hugging Face model: {e}")
            print(f"Full traceback for chat error with Hugging Face model:")
            traceback.print_exc()
            return "Sorry, I encountered an error while trying to process your request with the Llama model."


def main():
    """Main function to run the Streamlit chatbot."""
    

    try:
        me = Me()
    except Exception as e:
        st.error(f"Failed to initialize the application: {e}")
        traceback.print_exc()
        st.stop()

    st.title(f"{me.name} (Llama 3.1 via HF API)")

    st.write(f"Welcome! Ask me about {me.name}'s career, background, skills, and experience. This bot uses Llama 3.1 via Hugging Face Inference API.")
    #st.caption("Note: Tool usage (like saving email) is currently a placeholder and will be re-implemented for Llama 3.1.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message_item in st.session_state.messages:
        with st.chat_message(message_item["role"]):
            st.markdown(message_item["content"])

    if prompt := st.chat_input("What would you like to ask?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history_for_bot = [msg for msg in st.session_state.messages[:-1]]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking with Llama 3.1...")
            try:
                bot_response_text = me.chat(prompt, history_for_bot)
                message_placeholder.markdown(bot_response_text)
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                message_placeholder.error(error_message)
                traceback.print_exc()
                bot_response_text = "I'm having trouble connecting right now. Please try again later."
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response_text})

if __name__ == "__main__":
    main()