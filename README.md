# LoRA Fine-Tuning and CoreML Conversion with Streamlit

This project demonstrates how to fine-tune a large language model (LLM) using Low-Rank Adaptation (LoRA) and convert it to the CoreML format for on-device deployment. The entire process is wrapped in a user-friendly Streamlit web application.

## Features

-   **Fine-Tune LLMs with LoRA:** Easily fine-tune the `distilbert/distilgpt2` model on the `roneneldan/TinyStories` dataset using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.
-   **Text Generation:** Generate creative stories from a text prompt using the fine-tuned model.
-   **Adjustable Generation Parameters:** Control text generation with parameters like temperature, max length, and repetition penalty.
-   **CoreML Conversion:** Convert the fine-tuned model to a `.mlpackage` file, ready for integration into Apple ecosystem applications.
-   **Interactive Web UI:** A simple and interactive user interface built with Streamlit.

## How it Works

The application follows a simple workflow:

1.  **Load Base Model:** It starts by loading the pre-trained `distilbert/distilgpt2` model and its tokenizer from the Hugging Face Hub.
2.  **Fine-Tuning:** The user can initiate the fine-tuning process. The application uses the `peft` library to apply LoRA to the base model and trains it on the `TinyStories` dataset. The resulting LoRA adapter is saved locally.
3.  **Text Generation:** Once the model is fine-tuned (or a pre-existing adapter is loaded), you can provide a prompt to generate stories.
4.  **CoreML Conversion:** The application can merge the LoRA adapter with the base model and then convert the merged model into the CoreML format, which can be downloaded as a `.zip` file.

## Technologies Used

-   **Model:** `distilbert/distilgpt2` from Hugging Face
-   **Dataset:** `roneneldan/TinyStories` from Hugging Face
-   **Fine-Tuning:** `peft` (Parameter-Efficient Fine-Tuning) library with LoRA
-   **Framework:** PyTorch
-   **Web App:** Streamlit
-   **Model Conversion:** CoreMLTools
-   **Core Libraries:** `transformers`, `datasets`, `accelerate`, `sentencepiece`

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

4.  **Open the application in your browser:**
    Navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## File Structure

-   `app.py`: The main Python script containing the Streamlit application logic.
-   `requirements.txt`: A list of the Python packages required to run the project.
-   `README.md`: This file, providing information about the project.
-   `distilgpt2-lora-tinystories/`: (Generated Directory) This directory will be created to store the LoRA adapter after fine-tuning.
-   `results/`: (Generated Directory) This directory is used by the `transformers.Trainer` to save training outputs.
-   `distilgpt2-lora-tinystories.mlpackage/`: (Generated Directory) This directory will be created after the CoreML conversion.
-   `distilgpt2-lora-tinystories.zip`: (Generated File) The zipped CoreML model ready for download.

---

Happy fine-tuning and story generating! 