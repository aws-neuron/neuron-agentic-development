"""
Shared test infrastructure for {MODEL_NAME} component-level equivalence tests.
vLLM-Neuron variant — uses vLLM distributed init instead of NxDI.

Usage: Copy to {EXP_DIR}/tests/conftest.py, update constants.
"""
import os
import sys

os.environ["NXD_CPU_MODE"] = "1"
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "8099")
os.environ.setdefault("RANK", "0")

sys.path.insert(0, "{PORT_MODULE_DIR}")
sys.path.insert(0, "{SCRIPTS_DIR}")

import pytest
import torch
import numpy as np

# ── Constants (from model's config.json) ── UPDATE THESE
HIDDEN_SIZE = 0
NUM_HEADS = 0
NUM_KV_HEADS = 0
HEAD_DIM = 0
INTERMEDIATE_SIZE = 0
VOCAB_SIZE = 0
NUM_LAYERS = 0
RMS_NORM_EPS = 1e-6
BS = 1
SEQ_LEN = 8
VOCAB_SIZE_SMALL = 1024

TOLERANCE_RATIO = 1.2
MODEL_PATH = "{HF_MODEL_PATH}"

from tensor_compare import compare_3tensors, check_3tensor_result

# ── vLLM CPU-mode environment ──
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_model_parallel,
    destroy_distributed_environment,
)


def _init_cpu_env_tp1():
    init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    initialize_model_parallel(tensor_model_parallel_size=1)


def _destroy_mp():
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def init_cpu_env_session():
    _init_cpu_env_tp1()
    yield
    _destroy_mp()


@pytest.fixture(autouse=True)
def seed_every_test():
    torch.manual_seed(42)
    np.random.seed(42)
