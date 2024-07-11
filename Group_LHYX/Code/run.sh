export CUDA_VISIBLE_DEVICES=5

# [Baseline_vanilla_mlp, Baseline_concat_mlp, Baseline_SA, Ablation, OurModel]
python -u whole.py --dim 128 --depth 4 --epochs 500 --model_name "OurModel" >> log/our.txt 2>&1
python -u whole.py --dim 128 --depth 4 --epochs 500 --model_name "Ablation" >> log/ablation.txt 2>&1
python -u whole.py --dim 128 --depth 4 --epochs 500 --model_name "Baseline_vanilla_mlp" >> log/baseline_1.txt 2>&1
python -u whole.py --dim 128 --depth 4 --epochs 500 --model_name "Baseline_concat_mlp" >> log/baseline_2.txt 2>&1
python -u whole.py --dim 128 --depth 4 --epochs 500 --model_name "Baseline_SA" >> log/baseline_3.txt 2>&1


