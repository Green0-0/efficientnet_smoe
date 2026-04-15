# Immediate
- Double check the b0_deepmoe for correctness in the model architecture (I am using AI generated code that hot swaps the MBConv with a gated version, which should be superior to gating the conv outputs and matches the original sweep)
- Modify the training script/model loading script in b0_deepmoe to load the new model architecture, and train with the correct loss function
- Modify the hyperparameter sweeper to create a scoring function that trades sparsity for accuracy, and encourages 30%(?) sparsity and ~67% accuracy(?)
- Do b1, b2... etc versions if we have time

# Later
- Train the full models based on found hyperparameters on all the train/val data, and eval them on test
- Code a model analyzer that runs the model against the test script to determine if experts are specialized
- Manually calculate the FLOPs (note that we cannot programmatically calculate them because the sparsified layers still consume flops unless a custom sparse kernel is written to ignore them)

### Notes:
- Do NOT mix up the validation sets and the training sets, especially when doing hyperparameter sweeps