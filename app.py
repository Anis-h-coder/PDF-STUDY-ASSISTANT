import streamlit as st
from groq import Groq
import pdfplumber
import os
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client securely
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -------------------------------
# Function to generate response
# -------------------------------
def generate_response_groq(context, query):
    """Generate response using Groq API."""

    # Limit context size to avoid token overflow
    context = context[:3000]

    prompt = f"""
You are a helpful AI assistant.
Answer the question strictly based on the provided context.
If the answer is not in the context, say "The answer is not available in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.3
    )

    return chat_completion.choices[0].message.content


# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="PDF Query Application", layout="wide")

# Sidebar
st.sidebar.title("📄 PDF Query Assistant")
st.sidebar.image(
    "logo.png",
    use_container_width=True
)

st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Upload a PDF file  
2. Enter your question  
3. Click **Get Answer**
""")

st.sidebar.markdown("---")
st.sidebar.info("Developed using Streamlit and Groq API")


# Main Title
st.title("📄 PDF Query Application")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully! 📄")

    # Extract text
    document_text = extract_text_from_pdf(uploaded_file)

    # Show preview instead of full text
    st.subheader("📜 Extracted Text Preview")
    st.text_area("Preview", document_text[:1500], height=200)

    query = st.text_input("🔍 Enter your question")

    if st.button("💬 Get Answer"):
        if query:
            with st.spinner("Generating response..."):
                response = generate_response_groq(document_text, query)

            st.subheader("🧠 Answer")
            st.write(response)

            # Clear memory
            gc.collect()
        else:
            st.error("Please enter a question.")


# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Groq")
