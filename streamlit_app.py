import streamlit as st
import pandas as pd
from pdf2image import convert_from_bytes
import openai
import json
from typing import List, Dict
from pydantic import BaseModel, Field
import base64
from PIL import Image
import io
import time
import uuid

# Set page configuration
st.set_page_config(
    page_title="Invoice Data Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    .upload-success {
        color: #4CAF50;
        font-weight: bold;
    }
    .processing-complete {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def encode_image(image: Image.Image) -> str:
    """
    Encodes an image to a base64 string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

class InvoiceItem(BaseModel):
    """
    Represents an item in an invoice
    """
    reference_number: str | None = Field(None, description="The reference number of the item")
    serial_lot_number: str | None = Field(None, description="The serial or lot number of the item") 
    quantity: int = Field(..., description="The quantity of the item")

class Invoice(BaseModel):
    """
    Represents an invoice
    """
    items: List[InvoiceItem] = Field(..., description="The items in the invoice")

class Extractor:
    """
    Extracts information from an invoice image
    """

    def __init__(self, api_key: str = None):
        self.model_name: str = "gpt-4o"
        self.api_key = api_key
        self.client: openai.OpenAI = openai.OpenAI(
            api_key=self.api_key
        )
        self.prompt_template: str = """
            Your task is to extract all the items from an invoice
            The invoice is a list of items, each item has a reference number, serial/lot number and quantity
            Perform your task in these steps:
            1. Read the invoice image, and identify which values refer to required fields
            2. Extract the values from the image
            3. Return the values in a JSON format
        """

    def _format_images(self, images: List[Image.Image]) -> List[Dict[str, str]]:
        """
        Formats images for OpenAI API
        """
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(image)}"
                }
            }
            for image in images
        ]

    def extract_invoice(self, images: List[Image.Image]) -> dict:
        """
        Extracts information from an invoice image
        """

        response: openai.ChatCompletion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt_template
                        },
                        *self._format_images(images)
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

def process_pdf_file(filename, file_content, api_key):
    """
    Process a single PDF file and extract invoice data
    
    Args:
        filename: Name of the file
        file_content: Content of the file
        api_key: OpenAI API key
        
    Returns:
        Dictionary with filename and extracted content
    """
    try:
        # Convert PDF bytes to images
        images = convert_from_bytes(file_content)
        
        # Extract data
        extractor = Extractor(api_key=api_key)
        invoice_content = extractor.extract_invoice(images)
        
        return {
            'filename': filename,
            'content': invoice_content,
            'status': 'success'
        }
    except Exception as e:
        _ = f"Error processing {filename}: {str(e)}"
        return {
            'filename': filename,
            'content': json.dumps({"error": str(e)}),
            'status': 'error'
        }

def save_to_csv(data: List[Dict]) -> pd.DataFrame:
    """
    Converts extracted invoice data to a DataFrame
    
    Args:
        data: List of dictionaries containing invoice data
        
    Returns:
        DataFrame with extracted data
    """
    if not data:
        return pd.DataFrame()
    
    # Flatten the data structure for CSV
    flattened_data = []
    for invoice_dict in data:
        # Parse the JSON string if it's a string
        if isinstance(invoice_dict['content'], str):
            try:
                invoice_data = json.loads(invoice_dict['content'])
            except json.JSONDecodeError:
                continue
        else:
            invoice_data = invoice_dict['content']
        
        # Extract items from the invoice
        if 'items' in invoice_data and isinstance(invoice_data['items'], list):
            for item in invoice_data['items']:
                flattened_item = {
                    'filename': invoice_dict['filename'],
                    'reference_number': item.get('reference_number', ''),
                    'serial_lot_number': item.get('serial_lot_number', ''),
                    'quantity': item.get('quantity', '')
                }
                flattened_data.append(flattened_item)
        else:
            # Handle case where response doesn't match expected format
            flattened_item = {
                'filename': invoice_dict['filename'],
                'reference_number': '',
                'serial_lot_number': '',
                'quantity': '',
            }
            flattened_data.append(flattened_item)
    
    # Convert to DataFrame
    if flattened_data:
        df = pd.DataFrame(flattened_data)
        
        # Convert quantity to int where possible
        try:
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(df['quantity'])
            # Convert float quantities to int where possible
            mask = df['quantity'].apply(lambda x: isinstance(x, float) and x.is_integer())
            df.loc[mask, 'quantity'] = df.loc[mask, 'quantity'].astype(int)
        except Exception as e:
            print(f"Error converting quantity to int: {e}")
            pass
            
        return df
    else:
        return pd.DataFrame()

def get_csv_download_link(df, filename="invoice_data.csv"):
    """
    Generates a download link for a DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def main():
    st.title("Invoice Data Extractor")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key (optional)", type="password", 
                               help="Enter your OpenAI API key. If not provided, the app will use the default key.")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app extracts invoice data from PDF files using OpenAI's vision capabilities.
        
        Upload your invoice PDFs, and the app will extract:
        - Reference numbers
        - Serial/lot numbers
        - Quantities
        
        The extracted data will be available for download as a CSV file.
        """)
    
    # Main content
    st.header("Upload Invoice PDFs")
    
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files uploaded")
        
        # Create a unique session ID for this batch
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        
        # Process button
        if st.button("Extract Invoice Data"):
            # Progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process files sequentially
            results = []
            start_time = time.time()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Process the file
                result = process_pdf_file(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    api_key
                )
                
                results.append(result)
                
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Convert results to DataFrame
            df = save_to_csv(results)
            
            # Store results in session state
            st.session_state[f'results_{st.session_state.session_id}'] = {
                'df': df,
                'processing_time': processing_time,
                'num_files': len(uploaded_files),
                'successful': sum(1 for r in results if r['status'] == 'success'),
                'failed': sum(1 for r in results if r['status'] == 'error')
            }
            
            # Complete
            progress_bar.progress(100)
            status_text.text(f"Completed! Processed {len(uploaded_files)} files in {processing_time:.2f} seconds.")
    
    # Display results if available
    if 'session_id' in st.session_state and f'results_{st.session_state.session_id}' in st.session_state:
        results = st.session_state[f'results_{st.session_state.session_id}']
        df = results['df']
        
        st.markdown("---")
        st.header("Results")
        
        if not df.empty:
            # Display the data
            st.subheader("Extracted Data")
            st.dataframe(df)
            
            # Download link
            st.markdown(get_csv_download_link(df), unsafe_allow_html=True)
            
            # Export options
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="invoice_data.csv",
                mime="text/csv"
            )
            
            # Excel export option
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Invoice Data', index=False)
            
            st.download_button(
                label="Download Excel",
                data=buffer.getvalue(),
                file_name="invoice_data.xlsx",
                mime="application/vnd.ms-excel"
            )
        else:
            st.warning("No data was extracted from the invoices.")

if __name__ == "__main__":
    main() 