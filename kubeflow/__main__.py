from kubeflow.helper import DVICContainerOperation as ContainerOperation
from kubeflow.helper import DVICPipelineWrapper as PipelineWrapper
from kubeflow.helper import noop


DOCKER_IMG             = "yliess86/snook"

PIPELINE_NAME          = "Snook"
PIPELINE_DESC          = "Snook Training Experiments"

WORKING_DIR            = "/Snook"
BASE_PATH              = "/data"
MOUNT_PATH             = "/data/dl/Snook"

DATASET_PATH           = f"{BASE_PATH}/dataset"
TRAIN_SAMPLES          = 80_000
VALID_SAMPLES          = 10_000
TEST_SAMPLES           = 10_000
DATASET_GPU            =      1
DATASET_TILE           =      1

AUTOENCODER_EPOCHS     =  50
AUTOENCODER_REFINE     =   5
AUTOENCODER_BATCH_SIZE = 128
AUTOENCODER_N_WORKERS  =   8
AUTOENCODER_GPU        =   1
AUTOENCODER_SAVE       = f"{BASE_PATH}/autoencoder.ts"

CLASSIFIER_EPOCHS      =  10
CLASSIFIER_BATCH_SIZE  = 128
CLASSIFIER_N_WORKERS   =   8
CLASSIFIER_GPU         =   1
CLASSIFIER_SAVE        = f"{BASE_PATH}/classifier.ts"


def nameset(name: str, samples: int) -> ContainerOperation:
    return ContainerOperation(
        DOCKER_IMG,
        "-c",
        " ".join((
            f"PYTHONPATH={WORKING_DIR} python3 -m kubeflow.dataset",
            f"--samples {samples}",
            f"--type {name}",
            f"--dest {DATASET_PATH}",
            f"--tile {DATASET_TILE} ;",
            f"exit 0", #TODO: Fix when no more bpy segfault on exit
        )),
        name=f"{name}set",
    ).select_node().mount_host_path(BASE_PATH, MOUNT_PATH).gpu(DATASET_GPU)


with PipelineWrapper(PIPELINE_NAME, PIPELINE_DESC) as pipeline:
    trainset = nameset("train", TRAIN_SAMPLES)
    validset = nameset("valid", VALID_SAMPLES)
    testset  = nameset("test",  TEST_SAMPLES)
    dataset  = noop("dataset")

    trainset | dataset
    validset | dataset
    testset  | dataset


    autoencoder = ContainerOperation(
        DOCKER_IMG,
        "-c",
        " ".join((
            f"PYTHONPATH={WORKING_DIR} python3 -m kubeflow.autoencoder",
            f"--epochs {AUTOENCODER_EPOCHS}",
            f"--refine {AUTOENCODER_REFINE}",
            f"--batch_size {AUTOENCODER_BATCH_SIZE}",
            f"--n_workers {AUTOENCODER_N_WORKERS}",
            f"--dataset {DATASET_PATH}",
            f"--save {AUTOENCODER_SAVE} ;",
            f"exit 0", #TODO: Fix when no more bpy segfault on exit
        )),
        name="autoencoder",
    ).select_node().mount_host_path(BASE_PATH, MOUNT_PATH).gpu(AUTOENCODER_GPU)

    classifier = ContainerOperation(
        DOCKER_IMG,
        "-c",
        " ".join((
            f"PYTHONPATH={WORKING_DIR} python3 -m kubeflow.classifier",
            f"--epochs {CLASSIFIER_EPOCHS}",
            f"--batch_size {CLASSIFIER_BATCH_SIZE}",
            f"--n_workers {CLASSIFIER_N_WORKERS}",
            f"--dataset {DATASET_PATH}",
            f"--save {CLASSIFIER_SAVE} ;",
            f"exit 0", #TODO: Fix when no more bpy segfault on exit
        )),
        name="classifier",
    ).select_node().mount_host_path(BASE_PATH, MOUNT_PATH).gpu(CLASSIFIER_GPU)

    dataset | autoencoder
    dataset | classifier


    pipeline()