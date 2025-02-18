from math import floor
from time import sleep

import streamlit as st
import pandas as pd
from pathlib import Path
from google import genai
import os
from typing import List
import time
from dataclasses import dataclass

@dataclass
class AppConfig:
    CATEGORIES = [
        "Deep Learning",
        "Computer Vision",
        "Reinforcement Learning",
        "Natural Language Processing",
        "Optimization"
    ]

    PROMPT_TEMPLATE = """
    Function: Classify_Document
    Input: Research Paper (PDF/Text)
    
    Categories: {{
        {categories}
    }}
    Constraints : (
    Output: Return only two things, separated by a comma
    1. The exact category name from the list above, No other category, Just Text, No symbols etc,
    2. A brief reason for the classification
    )
    Example response format:
    Deep Learning, Uses neural networks for pattern recognition
    
    Constraints: Return only the category and reason, separated by a comma, without any additional text.
    """

class GeminiAPI:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-1.5-flash"

    def upload_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        #
        # with open(pdf_path, "rb") as file:
        #     pdf_content = file.read()
        uploaded_file = self.client.files.upload(
            file=pdf_path,
            config={'display_name': os.path.basename(pdf_path)}
        )
        return uploaded_file

    def process_pdf(self, pdf_path: str, prompt: str):
        uploaded_file = self.upload_pdf(pdf_path)
        # st.info(self.client.models.list(config={'page_size': 5, 'query_base' : True}).page)
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, uploaded_file]
        )
        return response.text if response else "No response received"

class DocHandler:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        self.pdf_files = self._get_pdf_files()

    def _get_pdf_files(self) -> List[Path]:
        return list(self.directory.glob('*.pdf'))

    def list_pdfs(self) -> List[Path]:
        return self.pdf_files

class CSVHandler:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df = pd.read_csv(file_path) if self.file_path.exists() else pd.DataFrame(columns=['Paper','Author', 'Year', 'PdfLink', 'Label', 'Reason'])

    def update_value(self, paper_name: str, label: str, reason: str) -> bool:
        try:
            mask = self.df['Paper'].str.contains(
                paper_name.replace('.pdf', '').strip(),
                case=False,
                na=False,
                regex=False
            )

            if not mask.any():
                # If paper not found, add new row
                new_row = pd.DataFrame({
                    'Paper': [paper_name],
                    'Author' : '',
                    'Year': '',
                    'PdfLink':'' ,
                    'Label': [str(label)],
                    'Reason': [str(reason)]
                })
                self.df = pd.concat([self.df, new_row], ignore_index=True)
            else:
                self.df.loc[mask, 'Label'] = str(label)
                self.df.loc[mask, 'Reason'] = str(reason)

            self.save_csv()
            return True

        except Exception as e:
            st.error(f"Error updating CSV for {paper_name}: {str(e)}")
            return False

    def save_csv(self):
        self.df.to_csv(self.file_path, index=False, encoding='utf-8')

class UI:
    def __init__(self):
        st.set_page_config(page_title="Research Docs Annotator", page_icon="📚")
        if 'processing' not in st.session_state:
            st.session_state.processing = False
    @staticmethod
    def render_header():
        st.title("📚 Research Docs Annotator")
        st.markdown("---")
    @staticmethod
    def get_multiple_inputs():
        total_cat = int(st.number_input('Number of Categories, 0 means default :', min_value=0, max_value=100, value=0, placeholder="Enter number of categories..."))
        inputs = []
        if total_cat < 1:
            return None
        for i in range(total_cat):
            input_string = st.text_input(f"Category : {i + 1}")
            inputs.append(input_string)
        return inputs
    @staticmethod
    def render_configuration():
        col1, col2 = st.columns(2)
        with col1:
            pdf_dir = st.text_input("PDF Directory Path",
                                    value="./downloaded_papers",
                                    help="Directory containing PDF files")
        with col2:
            csv_path = st.text_input("CSV Output Path",
                                     value="./metadata/papers_metadata.csv",
                                     help="Path to save classification results")

        api_key = st.text_input("Gemini API Key",
                                type="password",
                                help="Your Google Gemini API key")

        return pdf_dir, csv_path, api_key
    @staticmethod
    def render_progress( current, total ,progress_bar,progress_text,  current_info, info):
        with st.spinner("Please wait, Processing..."):
            progress = int(floor((current / total) * 100))
            progress_bar.progress(progress,text=f"{current} / {total} | {round((current / total) * 100, 2)}%")
            progress_text.text(f"Processing: {current}/{total} papers")
            current_info.code(info)
    @staticmethod
    def render_results( df):
        if df is not None and not df.empty:
            st.subheader("Classification Results")
            st.dataframe(df, use_container_width=True)

def main():
    ui = UI()
    ui.render_header()

    pdf_dir, csv_path, api_key = ui.render_configuration()
    categories_list = ui.get_multiple_inputs()
    if categories_list:
        AppConfig.CATEGORIES = categories_list
    if st.button("Start Classification", disabled=st.session_state.processing):
        if not api_key:
            st.error("Please enter your Gemini API Key")
            return

        try:
            st.session_state.processing = True

            # Initialize handlers
            doc_handler = DocHandler(pdf_dir)
            csv_handler = CSVHandler(csv_path)
            gemini_api = GeminiAPI(api_key)

            pdfs = doc_handler.list_pdfs()
            total_pdfs = len(pdfs)

            if total_pdfs == 0:
                st.warning(f"No PDF files found in {pdf_dir}")
                return

            st.info(f"Found {total_pdfs} PDFs to process")
            progress_bar = st.empty()
            progress_text = st.empty()
            current_info= st.empty()
            error_area =  st.empty()
            msg = ''
            error_info =''
            # Prepare prompt
            categories_str = ','.join(f'"{cat}"' for cat in AppConfig.CATEGORIES)
            prompt = AppConfig.PROMPT_TEMPLATE.format(categories=categories_str)

            # Process PDFs
            for idx, pdf in enumerate(pdfs, 1):
                try:
                    result = gemini_api.process_pdf(str(pdf), prompt)

                    if ',' not in result:
                        st.warning(f"Invalid response format for {pdf.name}: {result}")
                        continue

                    label, reason = result.split(',', 1)
                    label = label.strip()
                    reason = reason.strip()

                    if label not in AppConfig.CATEGORIES:
                        msg += f"{idx} : ❌Invalid category '{label}' for {pdf.name}\n"
                        continue

                    if csv_handler.update_value(pdf.name, label, reason):
                        msg += f'{idx} : ✓ Classified "{pdf.name}" as "{label}"\n'
                    else:
                        msg += f'{idx} : ⚠ CSV issue "{pdf.name}" as "{label}"\n'

                except Exception as e:
                    st.error(f'{idx} : ❌Error processing "{pdf.name}" : "{str(e)}", Trying next after few seconds')
                    error_info+=f'{idx} : Failed :  "{pdf.absolute()}"  > "{str(e)}"\n'
                    error_area.code(error_info)
                    sleep(10)
                    continue
                finally:

                    ui.render_progress(idx, total_pdfs,progress_bar, progress_text, current_info, msg )
                    time.sleep(0.1)  # Prevent UI freezing

            st.toast("Classification completed!", icon="✅")
            st.success("✅ Classification completed!")
            st.balloons()
            ui.render_results(csv_handler.df)

        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()