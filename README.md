# StructCoder

## Setup the conda enviroment:
conda create -n structcoder --file structcoder.yml <br>
conda activate structcoder

## Download pretrained checkpoint:
mkdir -p saved_models/pretrain/checkpoint-12000
Download the pretrained model weights from [here](https://drive.google.com/drive/folders/1cyvtmZjaLc1OwlnU0_N_GwC_eAs5snf9?usp=sharing) and place it under saved_models/pretrain/checkpoint-12000/

## Finetune on translation:
python3 run_translation.py --do_train --do_eval --do_test --source_lang java --target_lang cs --max_target_length 320 --alpha1_clip -4 --alpha2_clip -4 

python3 run_translation.py --do_train --do_eval --do_test --source_lang cs --target_lang java --max_target_length 256 --alpha1_clip -4 --alpha2_clip -4

## Finetune on text-to-code generation:
python3 run_generation.py --do_train --do_eval --do_test
