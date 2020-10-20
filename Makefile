# Flags
DEBUG=-d
CAMERA=2

# Third Party Libraries
VENDOR=./vendor
VENDOR_BPY=${VENDOR}/script/bpy_deps.sh
VENDOR_K210=${VENDOR}/script/k210_deps.sh

# Dataset Files
DATA=./resources/data
DATA_CONFIG=${DATA}/config.yaml
DATA_TRAIN=${DATA}/dataset/train
DATA_VALID=${DATA}/dataset/valid
DATA_TEST=${DATA}/dataset/test
DATA_TRAIN_SAMPLES=50_000
DATA_VALID_SAMPLES=500
DATA_TEST_SAMPLES=500

# Model Files
MODEL=./resources/model
MODEL_CONFIG=${MODEL}/config.yaml
MODEL_LOCNET_ONNX=${MODEL}/locnet.nx
MODEL_LOCNET_KMODEL=${MODEL}/locnet.kmodel
MODEL_LABELNET_ONNX=${MODEL}/labelnet.nx
MODEL_LABELNET_KMODEL=${MODEL}/labelnet.kmodel

# Python Path for Blender PyModule
PYTHONPATH=/usr/local/lib/python3.8/dist-packages/


all: vendor dataset train convert # benchmark

# Install Third Party Dependencies
vendor:
	sh ${VENDOR_BPY}
	sh ${VENDOR_K210}

# Generate Training Validation and Testing Dataset
dataset: dataset_train dataset_valid dataset_test
dataset_train:
	PYTHONPATH=${PYTHONPATH} python3 -m snook.data.generate -c ${DATA_CONFIG} -r ${DATA_TRAIN}/render -d ${DATA_TRAIN}/data -s ${DATA_TRAIN_SAMPLES}
dataset_valid:
	PYTHONPATH=${PYTHONPATH} python3 -m snook.data.generate -c ${DATA_CONFIG} -r ${DATA_VALID}/render -d ${DATA_VALID}/data -s ${DATA_VALID_SAMPLES}
dataset_test:
	PYTHONPATH=${PYTHONPATH} python3 -m snook.data.generate -c ${DATA_CONFIG} -r ${DATA_TEST}/render  -d ${DATA_TEST}/data  -s ${DATA_TEST_SAMPLES}

# Train Pytorch Model
train:
	python3 -m snook.model.train -c ${MODEL_CONFIG} -m ${MODEL} ${DEBUG}

# Convert Model as Torch, TorchScript, Onnx and Kmodel Files
convert:
	python3 -m snook.model.convert -c ${MODEL_CONFIG} -m ${MODEL} ${DEBUG}

	./vendor/bin/ncc compile ${MODEL_LOCNET_ONNX} ${MODEL_LOCNET_KMODEL} -i onnx -o kmodel -t k210
	./vendor/bin/ncc compile ${MODEL_LABELNET_ONNX} ${MODEL_LABELNET_KMODEL} -i onnx -o kmodel -t k210

# Benchmark Python Models
benchmark:
	python3 -m snook.model.benchmark -c ${MODEL_CONFIG} -m ${MODEL} -s 200

# Demo
demo:
	python3 -m snook.demo -m ${MODEL} -c ${CAMERA}