from kubeflow.helper import DVICContainerOperation as ContainerOperation
from kubeflow.helper import DVICPipelineWrapper as PipelineWrapper
from kubeflow.helper import noop


PIPELINE_NAME = "Snook"
PIPELINE_DESC = "Snook Training Experiments"

BASE_PATH     = "/data"
MOUNT_PATH    = "/data/dl/Snook"

DATASET_PATH  = f"{BASE_PATH}/dataset"
TRAIN_SAMPLES = 1_800_000
VALID_SAMPLES =   100_000
TEST_SAMPLES  =   100_000

with PipelineWrapper(PIPELINE_NAME, PIPELINE_DESC) as pipeline:
    dataset = ContainerOperation(
        "yhati/dataset",
        "keubeflow/dataset.py",
        f"--train {TRAIN_SAMPLES}",
        f"--valid {VALID_SAMPLES}",
        f"--test {TEST_SAMPLES}",
        f"--dest {DATASET_PATH}",
        name="dataset",
    ).select_node().mount_host_path(BASE_PATH, MOUNT_PATH)

    no_op = noop()
    pipeline()