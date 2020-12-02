from kubeflow.helper import DVICContainerOperation as ContainerOperation
from kubeflow.helper import DVICPipelineWrapper as PipelineWrapper
from kubeflow.helper import noop


PIPELINE_NAME          = "Snook"
PIPELINE_DESC          = "Snook Training Experiments"

BASE_PATH              = "/data"
MOUNT_PATH             = "/data/dl/Snook"

DATASET_PATH           = f"{BASE_PATH}/dataset"
TRAIN_SAMPLES          = 1_800_000
VALID_SAMPLES          =   100_000
TEST_SAMPLES           =   100_000

AUTOENCODER_EPOCHS     =  50
AUTOENCODER_REFINE     =   5
AUTOENCODER_BATCH_SIZE = 128
AUTOENCODER_N_WORKERS  =   8
AUTOENCODER_SAVE       = f"{BASE_PATH}/autoencoder.ts"

with PipelineWrapper(PIPELINE_NAME, PIPELINE_DESC) as pipeline:
    dataset = ContainerOperation(
        "yhati/dataset",
        "kubeflow/dataset.py",
        f"--train {TRAIN_SAMPLES}",
        f"--valid {VALID_SAMPLES}",
        f"--test {TEST_SAMPLES}",
        f"--dest {DATASET_PATH}",
        name="dataset",
    ).select_node().mount_host_path(BASE_PATH, MOUNT_PATH).gpu(0)

    autoencoder = ContainerOperation(
        "yhati/autoencoder",
        "kubeflow/autoencoder.py",
        f"--epochs {AUTOENCODER_EPOCHS}",
        f"--refine {AUTOENCODER_REFINE}",
        f"--batch_size {AUTOENCODER_BATCH_SIZE}",
        f"--n_workers {AUTOENCODER_N_WORKERS}",
        f"--dataset {DATASET_PATH}",
        f"--save {AUTOENCODER_SAVE}",
        name="autoencoder",
    ).select_node().mount_host_path(BASE_PATH, MOUNT_PATH).gpu(0)

    dataset | autoencoder

    no_op = noop()
    pipeline()