from nni.experiment import Experiment
from nni.experiment.config import ExperimentConfig

search_space = {
    'batch_size': {'_type': 'choice', '_value': [16, 32, 64]},
    'CTO': {'_type': 'choice', '_value': [32, 64, 128]},
}

experiment = Experiment('local')
# config = ExperimentConfig('local')

# So the model script is called *trial code*.
experiment.config.trial_command = 'python main.py --enable_nni True'
experiment.config.trial_code_directory = '.'

experiment.config.search_space = search_space

# %%
# Configure tuning algorithm
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we use :doc:`TPE tuner </hpo/tuners>`.
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'

# %%
# Configure how many trials to run
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here we evaluate 10 sets of hyperparameters in total, and concurrently evaluate 2 sets at a time.
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 1
experiment.config.trial_gpu_number = 1

experiment.config.training_service.use_active_gpu = True

# %%
# You may also set ``max_experiment_duration = '1h'`` to limit running time.
#
# If neither ``max_trial_number`` nor ``max_experiment_duration`` are set,
# the experiment will run forever until you press Ctrl-C.
#
# .. note::
#
#     ``max_trial_number`` is set to 10 here for a fast example.
#     In real world it should be set to a larger number.
#     With default config TPE tuner requires 20 trials to warm up.

# %%
# Step 4: Run the experiment
# --------------------------
# Now the experiment is ready. Choose a port and launch it. (Here we use port 8080.)
#
# You can use the web portal to view experiment status: http://localhost:8080.
experiment.run(8080)

# %%
# After the experiment is done
# ----------------------------
# Everything is done and it is safe to exit now. The following are optional.
#
# If you are using standard Python instead of Jupyter Notebook,
# you can add ``input()`` or ``signal.pause()`` to prevent Python from exiting,
# allowing you to view the web portal after the experiment is done.

# input('Press enter to quit')
experiment.stop()