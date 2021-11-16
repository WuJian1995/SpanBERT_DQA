#将hotpot_dev_distractor_v1.json和hotpot_train.v1.json放在data/下  https://hotpotqa.github.io/

#2wikimultihop_data 解压后放在data/下   https://github.com/Alab-NII/2wikimultihop
#并将train,dev改名为2wiki_train，2wiki_dev

#将2wiki 和hotpot数据集转换成squad的格式
python code/convert_hotpot2squad_new.py --data_dir data/2wikimultihop_data --task convert --data_type 2wiki

python code/convert_hotpot2squad_new.py --data_dir data --task convert --data_type hotpot

#train for hotpotqa onehop qa model训练一个hotpot的单挑模型
!python code/run_squad.py \
  --do_train \
  --do_eval \
  --model spanbert-base-cased \
  --train_file data/hotpot-all/train.json \
  --dev_file data/hotpot-all/dev.json \
  --train_batch_size 32 \
  --eval_batch_size 32  \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_metric f1 \
  --output_dir hotpot_output \
  --prefix dev_squad


#train for 2wikimultihop onehop qa system训练一个2wiki的单跳模型
!python code/run_squad.py \
  --do_train \
  --do_eval \
  --model spanbert-base-cased \
  --train_file data/hotpot-all/2wiki_train.json \
  --dev_file data/hotpot-all/2wiki_dev.json \
  --train_batch_size 32 \
  --eval_batch_size 32  \
  --learning_rate 2e-5 \
  --num_train_epochs 4 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_metric f1 \
  --output_dir 2wikimultihop_output \
  --prefix dev_squad

python code/run_squad.py \
  --do_eval \
  --model 2wikimultihop_output \
  --dev_file data/hotpot-all/2wiki_dev.json \
  --eval_batch_size 32  \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_metric f1 \
  --output_dir  2wikimultihop_output \
  --prefix dev

#分解2wiki的问题
python code/Question_Decomposition.py --data_type 2wiki_dev
python code/run_decomposition.py --task decompose --data_type 2wiki_dev

#预测2wiki单跳问题答案预测hotpot数据集一样的命令，改一下文件名即可
python code/run_squad.py \
  --do_eval \
  --model 2wikimultihop_output \
  --dev_file data/decomposed/2wiki_dev_b.1.json \
  --eval_batch_size 32  \
  --max_seq_length 512 \
  --n_best_size 4\
  --doc_stride 128 \
  --eval_metric f1 \
  --output_dir 2wikimultihop_output \
  --prefix dev_b_1_

python code/run_decomposition.py --task plug --data_type 2wiki_dev --topk 10

python code/run_squad.py \
  --do_eval \
  --model 2wikimultihop_output \
  --dev_file data/decomposed/converted_dev_b.2.json \
  --eval_batch_size 32  \
  --max_seq_length 512 \
  --doc_stride 128 \
  --eval_metric f1 \
  --output_dir 2wikimultihop_output \
  --prefix dev_b_2_