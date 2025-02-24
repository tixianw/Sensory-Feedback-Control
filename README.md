<div align=center>
  <h1>Sensory-Feedback-Control</h1>

![Python](https://img.shields.io/badge/Python-3776AB?logo=Python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=NumPy&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-00A3E0?logo=Numba&logoColor=white)
[![license: MIT](https://img.shields.io/badge/license-MIT-yellow)](https://opensource.org/licenses/MIT)

</div>

A package for applying sensory feedback control law for computational octopus arm model simulated in PyElastica.

## Dependency & installation

### Requirements
  - Python version: 3.9
  - Additional package dependencies include: [NumPy](https://numpy.org/doc/stable/user/absolute_beginners.html), [SciPy](https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide), [Numba](https://numba.readthedocs.io/en/stable/user/5minguide.html), [Matplotlib](https://matplotlib.org/stable/users/explain/quick_start.html), [tqdm](https://tqdm.github.io/), [PyElastica](https://github.com/GazzolaLab/PyElastica), [H5py](https://docs.h5py.org/en/stable/), and [Click](https://click.palletsprojects.com/en/stable/) (detailed in `pyproject.toml`)

### Installation

Before installation, create a Python virtual environment to manage dependencies and ensure a clean installation of the **Sensory-Feedback-Control** package.

1. Create and activate a virtual environment: (If you have already created a virtual environment for **Sensory-Feedback-Control**, directly activate it.)

    ```properties
    # Change directory to your working folder
    cd path_to_your_working_folder

    # Create a virtual environment of name `myenv`
    # with Python version 3.9
    conda create --name myenv python=3.9

    # Activate the virtual environment
    conda activate myenv

    # Note: Exit the virtual environment
    conda deactivate
    ```

2. Install Package: (two methods)

    ```properties
    ## Need ffmpeg installed from Conda
    conda install conda-forge::ffmpeg
    
    ## Install directly from GitHub
    pip install git+https://github.com/tixianw/Sensory-Feedback-Control.git

    ## Or clone and install
    git clone https://github.com/tixianw/Sensory-Feedback-Control.git (download directly if cannot clone)
    cd Sensory-Feedback-Control
    pip install .

<details>

<summary> Click me to expand/collapse developer environment setup </summary>

## Developer environment setup

1. Clone and install development dependencies:
    ```properties
    git clone [https://github.com/hanson-hschang/Signal-System.git](https://github.com/tixianw/Sensory-Feedback-Control.git)
    cd Sensory-Feedback-Control
    pip install pip-tools
    ```

2. Generate development requirements file:
    ```properties
    pip-compile pyproject.toml --output-file=requirements.txt
    ```

</details>

## Example

Please refer to [`examples`](https://github.com/tixianw/Sensory-Feedback-Control/tree/main/examples) directory and learn how to use this **Sensory-Feedback-Control** package. Two examples are provided:
  - [`BendFormation`](https://github.com/tixianw/Sensory-Feedback-Control/tree/main/examples/BendFormation) initialize a straight octopus arm and form a bend to reach a target.
  - [`BendPropagation`](https://github.com/tixianw/Sensory-Feedback-Control/tree/main/examples/BendPropagation) creates an octopus arm with initial bend and propagates it towards the target.

## License

This project is released under the [MIT License](https://github.com/tixianw/Sensory-Feedback-Control/blob/main/LICENSE).

## Contributing

1. Fork this repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m "feat: Add some amazing feature"`)
5. Push to the feature branch (`git push origin feat/amazing-feature`)
6. Open a Pull Request
