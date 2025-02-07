python src/inference.py --checkpoint_dir checkpoint-epoch1-chunk15 --tokenizer_path data/tokenizer/ --prompt "Once upon a time" --max_length 30 --temperature 0.8


python src/convert_to_hf.py --checkpoint_dir=checkpoint-epoch1-chunk15 --tokenizer_dir=data/tokenizer --config=config/config.yaml --output_dir=hf_model_output

python zero_to_fp32.py . ./converted_model
python src/inference.py --model_dir hf_model_output --prompt "Once upon a time" --max_length 100 --temperature 0.7 --top_p 0.9 --top_k 50 --repetition_penalty 1.2
