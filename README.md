# CLIP prefix REG

To train: 
- copy ```configuration_template.py``` into a new file ```configuration.py``` and update with custom settings
- Run ```train.py``` (arguments as specified in the file can be used to override settings on the fly)

To generate expressions:
- Run ```generate_reg.py``` with the ```--model_checkpoint``` argument pointing to the checkpoint file (other arguments as specified in the file can be used to override settings on the fly)
