# BI
usefull commands 
find . -type d -name "__pycache__" -exec rm -r {} +
export PYTHONPATH=$(pwd)
conda activate ml_env
python codes/step.py | tee logs/steps.log
python codes/Oraclefinetune.py | tee logs/OF.log
ssh jag@172.17.38.139
tmux attach -t step_train
tmux new -s step_train
tmux new -s face
tmux new -s classey
tmux new -s coco
tmux attach -t face
watch -n 1 nvidia-smi
python baseline/students.py | tee logs/student.log
python steps.py | tee logs/clpu.log
python steps.py | tee logs/er_ewc.log
python unlearn.py | tee l2ul.log
python steps.py | tee er_ewc_ga.log
python steps.py | tee er_ace_neggrad_plus.log
python steps.py | tee der_scrub.log

STEP_PATHS = {
    'step1': "//home/jag/codes/Bi/checkpoint.s/oracle/10_59.pth",  # PLACEHOLDER 1
    'step2': "/home/jag/codes/Bi/checkpoints/oracle/20_69.pth",  # PLACEHOLDER 2
    'step3': "/home/jag/codes/Bi/checkpoints/oracle/30_79.pth",  # PLACEHOLDER 3
    'step4': "/home/jag/codes/Bi/checkpoints/oracle/40_89.pth",  # PLACEHOLDER 4
    'step5': "/home/jag/codes/Bi/checkpoints/oracle/50_99.pth",  # PLACEHOLDER 5
}