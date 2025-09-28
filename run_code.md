# the sequential steps to run the code

the code is initially designed to run on the high-performance cluster ([Phoenix HPC](https://docs.pace.gatech.edu/phoenix_cluster/gettingstarted_phnx/)), and it is fully compatible with single-node execution or local machine execution.


## data generation

the code related to the data generation is in the `src/data` folder and also the `src/utils/data_utils.py` file. 

there are three steps to generate the data:

### define the constants

The `data_utils.py` file defines all the constants, which configures the Sionna simulator and the data generation process. It is critical to understand the constants and adjust them based on your own setting before running any codes. 


### generate the data

check the `scripts/data_gen_template.sh` file for running the data generation on the high-performance cluster. The key point is to set the suitable array size for the generation for training, regular testing and generalization testing datasets.  As shown below, the suitable array size will make the generation process parallelized and fasten the whole process. 

```bash
python3 -m src.data.generator.py --is_train              # Generate training data, typical array size is 1-9
python3 -m src.data.generator.py                         # Generate regular test data, typical array size is 1
python3 -m src.data.generator.py --is_gen                # Generate generalization test data, typical array size is 1-20
```

When in the single-node execution, it is recommended to use the `--debug` flag to generate the minimal dataset for each scenario (The size of the debug mode is also defined in the `data_utils.py` file). That is 

```bash
python3 -m src.data.generator.py --debug --is_train      # Debug mode: minimal training data
python3 -m src.data.generator.py --debug                 # Debug mode: minimal test data
python3 -m src.data.generator.py --debug --is_gen        # Debug mode: minimal generalization data
```


### obtain the normalization stats

an additional step to use the generated data is to obtain the normalization stats. the code related to the normalization stats is in the `src/utils/norm_utils.py` file. the normalization stats are obtained from the generated data. The code is shown below:
```bash
python3 -m src.utils.norm_utils
```
The normalization stats will be saved in the `z_refer/data/stats/[TDD/FDD]/normalization_stats.pkl` file.


## model training

the code related to the model training is in the `src/cp` folder which is based on the [PyTorch Lightning framework](https://lightning.ai/docs/pytorch/stable/) for the its flexibility and powerful features which integrates many essential features for the model training and evaluation. There are three critical steps to define a new model into the framework and training it.

### Define the model

the model should defined under the `src/cp/models` folder. It should inherit the `BaseCSIModel` class in the `src/cp/models/common/base.py` file. And the model should implement the `forward` method, which is the core of the model. after the model is defined, remember to register the model to the `PREDICTORS` class in the `src/cp/models/__init__.py` file. you can defined different model architectures for FDD and TDD scenarios, only need to register the model as different name in the `PREDICTORS` class. For details, please refer the `src/cp/models/__init__.py` and an example implementation of RNN model in the [`src/cp/models/baseline_models/rnn.py`](src/cp/models/baseline_models/rnn.py) file.

### config the training process

the training process is configured in the `src/cp/config/config.py` file. you need to carefully configure the dataset (training ratio, batch size, FDD or TDD, etc.), model (model name, model parameters, loaded checkpoints, etc.), training (number of epochs, gradient clip, early stopping, save checkpoint, etc.), optimizer (optimizer name, optimizer parameters, etc.), scheduler (scheduler name, scheduler parameters, etc.), loss (loss name, loss parameters, etc.). For details, please refer the `src/cp/config/config.py`. After the configuration is done, run
```bash
python3 -m src.cp.config.config --model [model_name] --output-dir [output_dir] --is_U2D [True/False] --config-file [yaml/json]
```
to store the configuration in the yaml or json file. The default output directory is `z_artifacts/config/cp/[model_name]/`. The sample configuration file for the RNN model in the TDD scenario is provided at `z_artifacts/config/cp/rnn/tdd_rnn.yaml`.


### train the model

run the following command to train the model. 
```bash
python3 -m src.cp.main --hparams_csi_pred [config_file]
```
Or if you want to submit the training jobs in the high-performance cluster, you can use the `scripts/cp_template.slurm` file.

The training log will be automatically saved in the output directory, the default directory is `z_artifacts/outputs/[TDD/FDD]/[model_name]/[date_time]/` and the checkpoint and tensorboard logs will be saved under the `ckpts` and `tb_logs` subfolders respectively. To check the training log, you can run the following command 
```bash
tensorboard --logdir [output_directory]/tb_logs
```


## noise degree testing

since the testing framework include many realistic noise types and those are not able to defined directly by the SNRs, we need to first testing the realistic noises' intensity and determine the suitable parameters for the realistic noise for the target testing SNRs. 

the code related to the noise degree testing is in the `src/noise` folder. The command
```bash
python3 -m src.noise.noise_degree
```
will generated the noise parameter mapping for the target testing SNRs. The mapping will be saved in the `z_artifacts/outputs/noise/noise_degree/[date_time]/decide_nd.json` file. A copy of this files is also stored at `src/noise/results/decide_nd.json` file for programmatic use.



## model testing

the model evaluation include many different parts: obtain the prediction performance and computational overhead. and visualize the results. The code related to the model evaluation is in the `src/testing` folder.

### config the testing process

the first and most important step is to config the model you want to involved in the testing and also make sure the checkpoint you want to evaluated is at the right places. The `src/testing/config.py` file is the configuration file for the testing process. and the path of the checkpoint should be conforming with the `get_ckpt_path` function in the `src/testing/get_models.py` file. The default checkpoint path is `z_artifacts/weights/[tdd/fdd]/[model_name]/model.ckpt`.


### obtain the computational overhead

the code related to the computational overhead is in the `src/testing/computational_overhead` folder. The code is shown below:
```bash
python3 -m src.testing.computational_overhead.main
```
The computational overhead will be saved in the `z_artifacts/outputs/testing/computational_overhead/[date_time]` folder, both the tdd checkpoint and fdd checkpoint for all the model configured in the `src/testing/config.py` file will be evaluated.


### obtain the prediction performance

the code related to the prediction performance is in the `src/testing/prediction_performance` folder.

It is recommended to use the SLURM array job to run the prediction performance testing since the thousands of scenarios are involved in the testing process and the single-node execution will be too slow. The script `scripts/testing_template.sh` is the template for the SLURM array job. The key point is to set the suitable array size for the testing for each model and the array size should be conform the `JOBS_PER_MODEL` in the `src/testing/config.py` file.

For the local execution, the command is shown below:
```bash
python3 -m src.testing.prediction_performance.main --model [model_name]
```
The prediction performance will be saved in the `z_artifacts/outputs/testing/prediction_performance/[model_name]/[full_test/slice_i]/[date_time]/` folder. If slurm array job is used, `slice_1~slice_JOBS_PER_MODEL` will be the slice directories, otherwise the local execution will be the `full_test` directory.


### handling the testing results

the code related to the handling the testing results is in the `src/testing/results` folder. run the following command will basically finish the three steps: check the completion status of the testing models, gather all the results and save the results to a csv file, and post-processing the results to obtain the scenario-wise distributions based on the NMSE and SE metrics.
```bash
python3 -m src.testing.results.main
```

the results will be saved in the `z_artifacts/outputs/testing/results/completion_reports/[date_time]/` folder, `z_artifacts/outputs/testing/results/gather/[date_time]/` folder, and `z_artifacts/outputs/testing/results/analysis/[nmse/se]/[date_time]/` folder, respectively.


### visualize the results

the code related to the visualization is in the `src/testing/vis` folder. there are four types of plots: line plot, radar plot, violin plot, and table. All operations are organized in the `main.py` file. The command is shown below:
```bash
python3 -m src.testing.vis.main
```
The visualization will be saved in the `z_artifacts/outputs/testing/vis/[date_time]/[line/radar/violin/table]/` folder.
