# Immediate
- Speedup model training more somehow if possible (Current time: 5 hrs/run)
- Increase sweep batch size range to [256, 512, 1024, 2048(?)], modify lr range accordingly
- Double check that the baseline properly implemented EfficientNet B0 at a smaller scale based on the tensorflow code
- Determine which layers of EfficientNet can be converted into MoE, and which layers we should try to convert into MoE, as well as the desired sparsity ratio
- Baseline hyperparameter sweeps for EfficientNet on iNaturalist, as well as the model modified with the adjustments from the CNN moe papers

# Later
- Implement a proper trainer for a MoE EfficientNet that supports different aux losses/load balancing techniques, different moe layer styles, shared layers, etc
- Manually sweep the model architecture on the small dataset, or use optuna to do NAS(?)
- Code a model analyzer that runs the model against the test script to determine if experts are specialized

# Possible
- Full model training of best hyperparameters on full iNaturalist paper, for both baseline, and desired MoE

### Notes:
- Do NOT mix up the validation sets and the training sets, especially when doing hyperparameter sweeps
- Do not make the model over 20% bigger (ie. the router)