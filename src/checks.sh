#!/bin/bash

# --- Calls the checking scripts --- #

# Call the venv
source loss_moments/bin/activate

# Call the main module
python3 -m src.loss_moments

# Call the unit tests
python3 -m pytest tests/ -s


echo "~~~ End of checks ~~~"