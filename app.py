from math import floor
from time import sleep
import streamlit as st
import pandas as pd
from pathlib import Path
import random
from google import genai
import os
from typing import List
import time
from dataclasses import dataclass
import json

@dataclass
class AppConfig:
    MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro"
    ]
    CATEGORIES = [
        "Deep Learning",
        "Computer Vision",
        "Machine Learning",
        "Natural Language Processing",
        "Optimization"
    ]
    PROMPT_TEMPLATE = """
Function: Classify_Document
Input: Research Paper Abstract (Text)

Categories: {{
    {categories}
}}
Constraints:
Output: Return a JSON object with exactly two keys: "category" and "reason".
- "category": The exact category name from the list above. even If the paper abstract doesn't match.
- "reason": A brief reason for the classification.
Example response format:
{{
    "category": "Deep Learning",
    "reason": "Uses neural networks for pattern recognition"
}}
"""

class GeminiAPI:
    def __init__(self, api_key: str, model_id: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

    def process_abstract(self, abstract: str, prompt: str):
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[prompt, abstract]
        )
        return response.text if response else "No response received"

class CSVHandler:
    def __init__(self, uploaded_file):
        self.df = pd.read_csv(uploaded_file)
        if 'Label' not in self.df.columns:
            self.df['Label'] = ''
        if 'Reason' not in self.df.columns:
            self.df['Reason'] = ''

    def update_value(self, paper_name: str, label: str, reason: str) -> bool:
        try:
            mask = self.df['Paper'].str.contains(
                paper_name.replace('.pdf', '').strip(),
                case=False,
                na=False,
                regex=False
            )
            if not mask.any():
                new_row = pd.DataFrame({
                    'Paper': [paper_name],
                    'Author': '',
                    'Year': '',
                    'PdfLink': '',
                    'Abstract': '',
                    'Label': [str(label)],
                    'Reason': [str(reason)]
                })
                self.df = pd.concat([self.df, new_row], ignore_index=True)
            else:
                self.df.loc[mask, 'Label'] = str(label)
                self.df.loc[mask, 'Reason'] = str(reason)
            return True
        except Exception as e:
            st.error(f"Error updating CSV for {paper_name}: {str(e)}")
            return False

    def get_dataframe(self):
        return self.df

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
        total_cat = int(st.number_input('Number of Categories, 0 means default:', min_value=0, max_value=100, value=0, placeholder="Enter number of categories..."))
        inputs = []
        if total_cat < 1:
            return None
        for i in range(total_cat):
            input_string = st.text_input(f"Category {i + 1}")
            inputs.append(input_string)
        return inputs

    @staticmethod
    def render_configuration():
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], help="Upload a CSV file containing paper metadata")
            model_options = AppConfig.MODELS
            model_selection = st.selectbox("Model Selection", model_options)
        with col2:
            api_key = st.text_input("Gemini API Key",
                                    type="password",
                                    help="Your Google Gemini API key")
        return uploaded_file, api_key, model_selection

    @staticmethod
    def render_progress(current, total, progress_bar, progress_text, current_info, info):
        with st.spinner("Please wait, Processing..."):
            progress = int(floor((current / total) * 100)) % 100
            progress_bar.progress(progress, text=f"{current} / {total} | {round((current / total) * 100, 2)}%")
            progress_text.text(f"Processing: {current}/{total} papers")
            current_info.code(info)

    @staticmethod
    def render_results(df):
        if df is not None and not df.empty:
            st.subheader("Classification Results (CSV)")
            st.dataframe(df, use_container_width=True)

def main():
    ui = UI()
    ui.render_header()
    delay = 60.0
    uploaded_file, api_key, model_selection = ui.render_configuration()
    categories_list = ui.get_multiple_inputs()
    if categories_list:
        AppConfig.CATEGORIES = categories_list

    if uploaded_file is not None and st.button("Start Classification", disabled=st.session_state.processing):
        if not api_key:
            st.error("Please enter your Gemini API Key")
            return

        try:
            st.session_state.processing = True

            # Initialize handlers
            csv_handler = CSVHandler(uploaded_file)
            gemini_api = GeminiAPI(api_key, model_id=random.choice(AppConfig.MODELS))

            df = csv_handler.get_dataframe()
            if 'Abstract' not in df.columns:
                st.error("The uploaded CSV file does not contain an 'Abstract' column.")
                return

            # Filter rows that have an abstract but no label
            rows_to_process = df[(df['Abstract'].notna()) & (df['Abstract'] != '')]
            total_rows = len(rows_to_process)

            if total_rows == 0:
                st.warning("No rows with abstracts and no labels found in the CSV file.")
                return

            st.info(f"Found {total_rows} rows to process")
            progress_bar = st.empty()
            progress_text = st.empty()
            current_info = st.empty()
            error_area = st.empty()
            msg = ''
            error_info = ''

            categories_str = ','.join(f'"{cat}"' for cat in AppConfig.CATEGORIES)
            prompt = AppConfig.PROMPT_TEMPLATE.format(categories=categories_str)

            for idx, row in rows_to_process.iterrows():
                try:
                    abstract = row['Abstract']
                    result = gemini_api.process_abstract(abstract, prompt)
                    result = result.strip()

                    # Remove Markdown formatting if present
                    if result.startswith("```") and result.endswith("```"):
                        lines = result.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        result = "\n".join(lines).strip()

                    try:
                        parsed_result = json.loads(result)
                        label = parsed_result.get("category", "").strip()
                        reason = parsed_result.get("reason", "").strip()
                    except json.JSONDecodeError:
                        st.warning(f"Invalid JSON response format for {row['Paper']}: {result}")
                        continue

                    if label not in AppConfig.CATEGORIES:
                        msg += f"{idx} : ❌ Invalid category '{label}' for {row['Paper']}\n"
                        label = 'Other'

                    if csv_handler.update_value(row['Paper'], label, reason):
                        msg += f'{idx} : ✓ Classified "{row["Paper"]}" as "{label}"\n'
                    else:
                        msg += f'{idx} : ⚠ CSV issue for "{row["Paper"]}"\n'

                except Exception as e:
                    st.error(f'{idx} : ❌ Error processing "{row["Paper"]}" : "{str(e)}", Trying next after few seconds')
                    error_info += f'{idx} : Failed: "{row["Paper"]}" > "{str(e)}"\n'
                    error_area.code(error_info)
                    gemini_api = GeminiAPI(api_key, model_id=random.choice(AppConfig.MODELS))
                    sleep(delay)
                    continue
                finally:
                    ui.render_progress(idx, total_rows, progress_bar, progress_text, current_info, msg)
                    time.sleep(0.1)

            st.toast("Classification completed!", icon="✅")
            st.success("✅ Classification completed!")
            st.balloons()
            ui.render_results(csv_handler.get_dataframe())

        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            st.session_state.processing = False

if __name__ == "__main__":
    main()