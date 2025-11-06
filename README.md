# Pytorch Deep Learning
This project has my notebooks from the Daniel Bourke [Learn PyTorch for Deep Learning](https://www.learnpytorch.io/) course and video [Learn PyTorch for deep learning in a day](https://www.youtube.com/watch?v=Z_ikDlimN6A).
I've also made life difficult for myself by using AMD ROCm instead of Nvidia cards, and have created a [dev container](https://devcontainers.github.io) that uses the AMD ROCm PyTorch docker container.

## Setup
### Prerequisites
You'll need the host machine where docker is running to have the AMD drivers installed, I had already set up ROCm as well before trying the dev containers, so I'm not sure if that's needed or not.
In my case I went to the [AMD Radeon Pro Linux driver](https://www.amd.com/en/support/download/linux-drivers.html) page and found the section "*AMD ROCm™ 7.1 for Ubuntu 24.04.3 HWE, supporting AMD Radeon™ Graphics Preview*"
### Automatic steps
Currently the easiest way is to use the dev container in the .devcontainer folder for dev work. This has been created for my IntelliJ environment, but should be modifiable to the IDE of your choice as dev containers is a common technique and is supported by many IDEs out of the box.
The dev container folder has the dev container definition, the docker compose file (there are container run time parameters needed), a .env file that should contain the UID and GID of the user in the host system, and a script that run post container attchment, that takes the AMD provided python venv, with the torch/rocm libaries, and creates a local user varient so other requirements can be installed from the `requirements.txt` on container startup or from the command line. IntelliJ also does a `pip install` for it's [Jupyter notebook](https://jupyter.org/) plugin installer.
### Manual steps
The down side of the venv setup is that it doesn't exist at the time the IDE attaches to it (the script is run post attachment as it then runs as the correct user, otherwise the venv gets created as root and the problem of permissions persists). So post attachment for the first time, the venv will need to be created as an SDK in the IDE and then assigned to the project as the python interpreter.

## Testing
There is a `torch-cuda-test.py` test script that will check for PyTorch acceleration and report the list of accelerators that can be used. This should confirm the environment is configured correctly.

## Notebooks
These are the notebooks developed during the course.
