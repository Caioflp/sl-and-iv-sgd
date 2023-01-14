# Statistical Learning and Inverse Problems: A Stochastic Gradient Descent Approach

Code Implementations for the paper 'Statistical Learning and Inverse
Problems: A Stochastic Gradient Descent Approach', published in NeurIPS
2022.

## Repository structure

* `res`: Resources directory. Contains useful articles and books.
* `src`: Where all code is located.
* `src/main.py`: Where all tweaking and thumbling around takes place.
* `src/tests.py`: Tests for some code in `code/src`.
* `src/utils`: Utilities separated by subject.
* `src/data`: Where data is stored. Also contains modules for interacting
  with said data.
* `src/experiments.py`: Code for experiments.
* `src/models.py`: Implementations of models.
* `src/losses.py`: Implementations of loss functions.
* `src/runs`: Folder where experiment output is stored. This includes
  trained models, visualizations as well as log files.

After running an experiment (with `exp.run()`) from `src/main.py`, its
output will be located at `src/runs/<experiment ID>`, where
`experiment ID` is the date, including hour, minute and second, when it
as run.

*Make sure you add the root directory directory to `PYTHONPATH` before running
`src/main.py`.* For example, from within `src/` you may run
```sh
    PYTHONPATH=..:$PYTHONPATH python main.py
```

Alternitively, you may run `main.py` as a package, using the following command from the repository root:
```sh
  python -m src.main
```
