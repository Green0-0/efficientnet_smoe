# Folders
- papers_list includes a list of all relevant papers I could find that applies a MoE to a CNN layer
- put chatgpt generated code in ai_slop
- final code will go in scripts
- ``pip install uv``, ``uv venv --python 3.12``, ``uv pip install -r requirements.txt`` :)

# Problems (so many)
- The project lists a bunch of papers about MoE and vMOE
    - None of those papers are applicable to a CNN (EfficientNet B0)
- The iNaturalist dataset is 300gb (wtf) and training EfficientNet on it will take 5-9 days (wtf)
    * Note: If we implement a multigpu trainer to split the training across 4 gpus it might be doable in 3 ish days instead.