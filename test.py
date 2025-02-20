update this code, replace by pdf to by abstract:
again go trought reqs : i need json, modify this code part of gemini, and  also fix issues:
Remove pdf functionality, just give abstract and get category and add a column in this papers_metadata.csv file, name Label, Reason, Keep existing values and feilds if they exist, Get the Label for row (using its abstract) if only if it has no label in file for that row, otherwise skip, and if absract not exist in csv , in any case, no abstract columnexist, Just prompt to put csv file location, with an info

Paper,Author,Year,PdfLink,Abstract,Label,reason
Modeling the Olfactory Bulb - Coupled Nonlinear Oscillators,"Zhaoping Li, John J. Hopfield",1988,https://papers.nips.cc/paper_files/paper/1988/file/013d407166ec4fa56eb1e1f8cbe183b9-Paper.pdf,"The olfactory  bulb of mammals  aids  in  the  discrimination  of  odors.  A  mathematical  model  based  on  the  bulbar  anatomy  and  electrophysiology  is  described.  Simulations  produce  a  35-60  Hz  modulated activity coherent across the bulb, mimicing the observed  field  potentials.  The  decision  states  (for  the  odor  information)  here  can  be  thought  of as  stable  cycles,  rather  than  point  stable  states  typical  of simpler  neuro-computing  models.  Analysis  and  simulations show that a  group of coupled non-linear oscillators are  responsible for the oscillatory activities determined by the odor in(cid:173) put, and that the bulb, with appropriate inputs from higher centers,  can  enhance  or suppress  the  sensitivity  to  partiCUlar  odors.  The  model provides a framework  in which to understand the transform  between odor input and the bulbar output to olfactory cortex.
"
Efficient Parallel Learning Algorithms for Neural Networks,"Alan Kramer, Alberto Sangiovanni-Vincentelli",1988,https://papers.nips.cc/paper_files/paper/1988/file/02522a2b2726fb0a03bb19f2d8d9524d-Paper.pdf,"Parallelizable optimization techniques are applied to the problem of  learning in feedforward neural networks. In addition to having supe(cid:173) rior convergence properties, optimization techniques such as the Polak(cid:173) Ribiere method are also significantly more efficient than the Back(cid:173) propagation algorithm. These results are based on experiments per(cid:173) formed on small boolean learning problems and the noisy real-valued  learning problem of hand-written character recognition.
"
The Boltzmann Perceptron Network_ A Multi-Layered Feed-Forward Network Equivalent to the Boltzmann Machine,"Eyal Yair, Allen Gersho",1988,https://papers.nips.cc/paper_files/paper/1988/file/045117b0e0a11a242b9765e79cbf113f-Paper.pdf,"The  concept  of  the  stochastic  Boltzmann  machine  (BM)  is  auractive  for  decision  making  and  pattern  classification  purposes  since  the  probability  of  attaining  the network  states  is a  function  of the network energy.  Hence,  the  probability of attaining particular energy minima  may be associated  with  the  probabilities  of  making  certain  decisions  (or  classifications).  However,  because of its stochastic  nature,  the complexity of the BM is fairly  high and  therefore  such  networks  are  not  very  likely  to  be  used  in  practice.  In  this  paper  we  suggest  a  way  to  alleviate  this  drawback  by  converting  the  sto(cid:173) chastic  BM into  a  deterministic  network  which  we  call  the  Boltzmann  Per(cid:173) ceptron  Network  (BPN).  The BPN is functionally  equivalent  to  the  BM but  has  a  feed-forward  structure  and  low  complexity.  No annealing  is required.  The  conditions  under  which  such  a  convmion  is  feasible  are  given.  A  learning  algorithm  for  the  BPN based  on  the  conjugate  gradient  method  is  also provided which is somewhat akin  to the backpropagation algorithm.
"
Computer Modeling of Associative Learning,"Daniel Alkon, Francis Quek, Thomas Vogl",1988,https://papers.nips.cc/paper_files/paper/1988/file/069059b7ef840f0c74a814ec9237b6ec-Paper.pdf,Abstract Unavailable
Neural Networks that Learn to Discriminate Similar Kanji Characters,"Yoshihiro Mori, Kazuhiko Yokosawa",1988,https://papers.nips.cc/paper_files/paper/1988/file/06409663226af2f3114485aa4e0a23b4-Paper.pdf,"is
"

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

    def upload_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        uploaded_file = self.client.files.upload(
            file=pdf_path,
            config={'display_name': os.path.basename(pdf_path)}
        )
        return uploaded_file

    def process_pdf(self, pdf_path: str, prompt: str):
        uploaded_file = self.upload_pdf(pdf_path)
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
                new_row = pd.DataFrame({
                    'Paper': [paper_name],
                    'Author': '',
                    'Year': '',
                    'PdfLink': '',
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
            pdf_dir = st.text_input("PDF Directory Path",
                                    value="./downloaded_papers",
                                    help="Directory containing PDF files")
            model_options = AppConfig.MODELS

            model_selection = st.selectbox("Model Selection", model_options)
        with col2:
            csv_path = st.text_input("CSV Output Path",
                                     value="./metadata/papers_metadata.csv",
                                     help="Path to save classification results")
        api_key = st.text_input("Gemini API Key",
                                type="password",
                                help="Your Google Gemini API key")
        return pdf_dir, csv_path, api_key, model_selection

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
    pdf_dir, csv_path, api_key, model_selection = ui.render_configuration()
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
            gemini_api = GeminiAPI(api_key, model_id=random.choice(AppConfig.MODELS))

            pdfs = doc_handler.list_pdfs()
            total_pdfs = len(pdfs)

            if total_pdfs == 0:
                st.warning(f"No PDF files found in {pdf_dir}")
                return

            st.info(f"Found {total_pdfs} PDFs to process")
            progress_bar = st.empty()
            progress_text = st.empty()
            current_info = st.empty()
            error_area = st.empty()
            msg = ''
            error_info = ''

            categories_str = ','.join(f'"{cat}"' for cat in AppConfig.CATEGORIES)
            prompt = AppConfig.PROMPT_TEMPLATE.format(categories=categories_str)

            for idx, pdf in enumerate(pdfs, 1):
                try:
                    result = gemini_api.process_pdf(str(pdf), prompt)
                    result = result.strip()
                    # Remove Markdown formatting if present
                    if result.startswith("
                                         ") and result.endswith("
                                         "):
                    lines = result.splitlines()
                    if lines[0].startswith("
                    "):
                    lines = lines[1:]
                    if lines and lines[-1].startswith("
                                                      "):
                    lines = lines[:-1]
                    result = "\n".join(lines).strip()

                    try:
                        parsed_result = json.loads(result)
                        label = parsed_result.get("category", "").strip()
                        reason = parsed_result.get("reason", "").strip()
                    except json.JSONDecodeError:
                        st.warning(f"Invalid JSON response format for {pdf.name}: {result}")
                        continue

                    if label not in AppConfig.CATEGORIES:
                        msg += f"{idx} : ❌ Invalid category '{label}' for {pdf.name}\n"
                        pdfs.append(pdf)
                        continue

                    if csv_handler.update_value(pdf.name, label, reason):
                        msg += f'{idx} : ✓ Classified "{pdf.name}" as "{label}"\n'
                    else:
                        msg += f'{idx} : ⚠ CSV issue for "{pdf.name}"\n'

                except Exception as e:
                    st.error(f'{idx} : ❌ Error processing "{pdf.name}" : "{str(e)}", Trying next after few seconds')
                    error_info += f'{idx} : Failed: "{pdf.absolute()}" > "{str(e)}"\n'
                    error_area.code(error_info)
                    gemini_api = GeminiAPI(api_key, model_id=random.choice(AppConfig.MODELS))
                    pdfs.append(pdf)
                    sleep(delay)
                    continue
                finally:
                    ui.render_progress(idx, len(pdfs), progress_bar, progress_text, current_info, msg)
                    time.sleep(0.1)
                    if len(pdfs) > floor(total_pdfs + total_pdfs * 0.25):
                        break;

            csv_handler.save_csv()
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

use latest web docs for geimin json output, leth the gemini api decide label