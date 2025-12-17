## Supervised Finetuning Instructions

- Preprocessing: Obtain the [TxT360-3efforts](https://huggingface.co/datasets/LLM360/TxT360-3efforts) dataset, and run `create_sft_mix_3effort.py` to get the weighted dataset. The weights are specified in `mix_3efforts.json`. The final data will be output to `YOUR_OUTPUT_DIRECTORY` (feel free to change this in the script)
- Run the SFT script.
