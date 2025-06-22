import streamlit as st
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
import coremltools as ct
import os
import zipfile
import tempfile

MODEL_NAME = "distilbert/distilgpt2"
DATASET_NAME = "roneneldan/TinyStories"
ADAPTER_PATH = "distilgpt2-lora-tinystories"

@st.cache_resource
def load_base_model_and_tokenizer():
    """Loads the base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return model, tokenizer

def load_and_prepare_dataset(tokenizer, split="train"):
    """Loads and tokenizes the dataset."""
    dataset = load_dataset(DATASET_NAME, split=split)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=256
        )
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Handle different dataset types safely
    try:
        if hasattr(dataset, 'column_names'):
            remove_cols = dataset.column_names
        else:
            remove_cols = None
    except:
        remove_cols = None

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=remove_cols
    )
    return tokenized_dataset

def fine_tune_model(model, tokenizer, tokenized_dataset):
    """Fine-tunes the model using LoRA."""
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=0.5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=100,
        eval_steps=50,
        warmup_steps=10,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        max_steps=100,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    peft_model.save_pretrained(ADAPTER_PATH)
    return peft_model

def convert_to_coreml(model, tokenizer):
    """Converts the model to CoreML format."""
    st.info("Merging LoRA adapter...")
    merged_model = model.merge_and_unload()
    st.success("Adapter merged.")

    st.info("Moving model to CPU for CoreML conversion...")
    merged_model = merged_model.cpu()
    merged_model.eval()
    st.success("Model moved to CPU.")

    # Create a simple wrapper that only returns logits
    class SimpleModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids):
            outputs = self.model(input_ids)
            return outputs.logits

    simple_model = SimpleModel(merged_model)
    st.info("Created simple model wrapper.")

    st.info("Tracing the model...")
    example_input = tokenizer("Once upon a time", return_tensors="pt")
    input_ids = example_input.input_ids
    
    # Ensure input is on CPU
    input_ids = input_ids.cpu()
    
    with torch.no_grad():
        traced_model = torch.jit.trace(simple_model, input_ids)
    st.success("Model traced.")

    st.info("Converting to CoreML ML Program...")
    coreml_model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_ids", shape=(1, 512), dtype=int)],
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    st.success("Conversion to CoreML complete.")

    output_path = f"{ADAPTER_PATH}.mlpackage"
    # Save CoreML model using the correct method
    try:
        coreml_model.save(output_path)
    except AttributeError:
        # Alternative method for newer versions
        ct.models.MLModel(coreml_model).save(output_path)
    return output_path


def main():
    st.title("LoRA Fine-Tuning of distilgpt2 for TinyStories")
    st.write("This app fine-tunes the `distilbert/distilgpt2` model on the `TinyStories` dataset using LoRA and PEFT.")

    # --- Load Model and Tokenizer ---
    with st.spinner("Loading base model and tokenizer..."):
        base_model, tokenizer = load_base_model_and_tokenizer()
        st.session_state.base_model = base_model
        st.session_state.tokenizer = tokenizer
    st.success("Base model and tokenizer loaded.")
    st.markdown(f"**Model:** `{MODEL_NAME}`")

    # --- Fine-Tuning ---
    st.header("1. LoRA Fine-Tuning")
    if st.button("Start Fine-Tuning"):
        with st.spinner("Loading dataset and fine-tuning... This might take a few minutes."):
            tokenized_dataset = load_and_prepare_dataset(tokenizer)
            st.session_state.tokenized_dataset = tokenized_dataset
            
            # Safe way to get dataset length
            try:
                dataset_length = len(tokenized_dataset)
                st.info(f"Dataset loaded with {dataset_length} examples.")
            except (TypeError, AttributeError):
                st.info("Dataset loaded (length unknown).")
            
            peft_model = fine_tune_model(base_model, tokenizer, tokenized_dataset)
            st.session_state.peft_model = peft_model
            st.success("Fine-tuning complete! LoRA adapter saved.")
            st.balloons()
    
    # Check if adapter exists to offer loading it
    if os.path.exists(ADAPTER_PATH) and "peft_model" not in st.session_state:
        if st.button("Load Fine-Tuned LoRA Adapter"):
            with st.spinner("Loading fine-tuned model..."):
                peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
                st.session_state.peft_model = peft_model
                st.success("Fine-tuned LoRA model loaded.")


    # --- Text Generation ---
    if "peft_model" in st.session_state:
        st.header("2. Generate Story")
        prompt = st.text_input("Enter a prompt to start a story:", "Once upon a time, in a land full of sunshine,")
        
        # Generation parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        with col2:
            max_length = st.slider("Max Length", 50, 200, 100, 10)
        with col3:
            repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.2, 0.1)
        
        if st.button("Generate"):
            with st.spinner("Generating text..."):
                model = st.session_state.peft_model
                inputs = tokenizer(prompt, return_tensors="pt")
                
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model.generate(
                    **inputs, 
                    max_length=max_length, 
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.write("### Generated Story:")
                st.write(generated_text)

        # --- CoreML Conversion ---
        st.header("3. Convert to CoreML")
        if st.button("Convert Model to CoreML"):
            with st.spinner("Converting model to CoreML format..."):
                coreml_model_path = convert_to_coreml(st.session_state.peft_model, st.session_state.tokenizer)
                st.success(f"Model successfully converted and saved to `{coreml_model_path}`")
                
                # For .mlpackage files, we need to create a zip file for download
                zip_path = f"{ADAPTER_PATH}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(coreml_model_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, coreml_model_path)
                            zipf.write(file_path, arcname)
                
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="Download CoreML Model",
                        data=f,
                        file_name=os.path.basename(zip_path),
                        mime="application/zip"
                    )

if __name__ == "__main__":
    main() 