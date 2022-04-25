# Bugs

**Expected behavior**: SA-PINO with self-adaptive learning rate of 0 should find the same solution as PINO. 

**Bug**: We find significantly different solutions between the two models.
See the results here: https://wandb.ai/rishigundakaram/sa-pino/groups/NS-direct/workspace?workspace=user-rishigundakaram. In particular there is a large tradeoff between IC error and Functional Error. 

**Repeatable Behavior**: Run the following to get the same results: 
```
python3 pathToPINOCode/run_pino3d_sa.py pathToPINOCode/debugging/NS-direct-PINO.yaml

python3 pathToPINOCode/run_pino3d_sa.py pathToPINOCode/debugging/NS-direct-SA-PINO-lr-0.yaml
```

**Tested behavior**
1. Weights in SA-PINO are indeed all 1 in between iterations.
2. Yaml files are the same except for the self_adaptive flag.
3. Normalization is not responsible for the behavior.
4. Seeds are the same.
5. Checked that the gradients are the same for first iteration. (indirect.txt has the gradients for the SA-PINO, and direct.txt has the gradients for the PINO). 