dataset = 'vasko'
batch_size = 4
block_size = 256 
gradient_accumulation_steps = 16 # acum so batch size is 64, block size is 256, tokens per iter = 16,384; batch and block size values are taken from tiny shaekspere config 
eval_iters = 400 # val split 400,000~ tokens: 400,000 / (4*256) ~= 390, if we consider uniform sampling in get_batch, this should approx. cover all of validatoin set in tokens 
max_iters = 600 # train_split 3,583,645 tokens: 3,583,645 / (4 * 16 * 256) ~= 219, we train slightly less than 3 epochs 
warmup_iters = 30 # 0.05 of max_iters = 600 * 0.05 = 30
eval_interval = 50 
compile = False
