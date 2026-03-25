# Quickstart

## 1. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Generate or refresh the dataset

```bash
python dataset-generator.py
```

## 3. Choose a base model

The repo now defaults to `Qwen/Qwen2.5-0.5B-Instruct`, which is a public model and much smaller than a 7B checkpoint.

If you want a different model:

```bash
export BASE_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

If the model is gated or private:

```bash
hf auth login
export HF_TOKEN=your_token_here
```

## 4. Train

```bash
python train.py
```

The merged fine-tuned model will be written to `qwen-calculus-finetuned/`.

## 5. Run the app

```bash
python app.py
```

Then open `http://127.0.0.1:7860`.

## Optional: Docker

Because this repo currently uses a lowercase `dockerfile`, build with:

```bash
docker build -f dockerfile -t calculus-tutor .
docker run --rm -p 7860:7860 calculus-tutor
```

## Notes

- On CPU, training will be much slower than on a GPU instance.
- If you switch to a larger base model, you may need to reduce `per_device_train_batch_size` in `train.py`.
- If Hugging Face download errors persist, double-check `BASE_MODEL_NAME` and authentication first.
