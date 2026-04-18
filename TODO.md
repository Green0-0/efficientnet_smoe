# Immediate
- Double check the deepmoe for correctness in the model architecture (I am using AI generated code that hot swaps the MBConv with a gated version, which should be superior to gating the conv outputs and matches the original sweep), and determine where the gate should be
- Add the second (rather, third) stage of training where the embedding is frozen, for some number of epochs
- Accurate external FLOP profiler, calculate the number of FLOPs in the efficientnet
- Sweep ReLU init value

# Later
- Train the full models based on found hyperparameters on all the train/val data, and eval them on test
- Code a model analyzer that runs the model against the test script to determine if experts are specialized
- Manually calculate the FLOPs (note that we cannot programmatically calculate them because the sparsified layers still consume flops unless a custom sparse kernel is written to ignore them)

### Notes:
- Do NOT mix up the validation sets and the training sets, especially when doing hyperparameter sweeps