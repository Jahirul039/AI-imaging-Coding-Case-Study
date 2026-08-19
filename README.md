# AI-imaging-Coding-Case-Study
raw image → segmentation → quantitative region features → structured JSON record → narrative
All models run locally via Ollama. Every number in the output record is computed in
Python and injected — the LLM only picks labels from a closed vocabulary and writes prose.
Educational use only; not cleared for clinical use.
Files
	•	biomedical_pipeline.ipynb — full pipeline (Tasks 1–4 + robustness and loss-ablation extensions)
	•	nuclei_dataset.zip — 112 images + 4 corrupted variants
	•	report.pdf — 4-page write-up
 unzip nuclei_dataset.zip
pip install torch torchvision scikit-image matplotlib pandas pillow scipy

ollama serve                     # separate terminal
ollama pull llama3.2-vision
ollama pull llava:7b             # fallback, selected automatically if the above fails
ollama pull llama3.1:8b
Results
|Model            |Val Dice  |Val IoU   |Test Dice |Test IoU  |
|-----------------|----------|----------|----------|----------|
|Otsu + morphology|0.9742    |0.9497    |0.9749    |0.9511    |
|**U-Net (BCE)**  |**0.9945**|**0.9891**|**0.9940**|**0.9881**|

Loss ablation: BCE 0.9945, BCE+Dice 0.9930, soft Dice 0.9925.
Three findings: object count MAE is 6.0 despite Dice 0.994, since a binary mask cannot
separate touching nuclei; three identical VLM calls gave three different descriptions,
all at confidence: high; and under low contrast the U-Net collapses to Dice 0.046
while Otsu holds at 0.957.
 
