# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import json

from braket.device_schema import (
    OpenQASMDeviceActionProperties,
    OpenQASMProgramSetDeviceActionProperties,
)
from braket.task_result import GateModelTaskResult
from braket.tasks import GateModelQuantumTaskResult

ACTION_PROPERTIES = OpenQASMDeviceActionProperties.parse_raw(
    json.dumps(
        {
            "actionType": "braket.ir.openqasm.program",
            "version": ["1"],
            "supportedOperations": ["rx", "ry", "h", "cy", "cnot", "unitary"],
            "supportedResultTypes": [
                {
                    "name": "StateVector",
                    "observables": None,
                    "minShots": 0,
                    "maxShots": 0,
                },
                {
                    "name": "AdjointGradient",
                    "observables": ["x", "y", "z", "h", "i", "hermitian"],
                    "minShots": 0,
                    "maxShots": 0,
                },
            ],
        }
    )
)

ACTION_PROPERTIES_PROGRAMSET = OpenQASMProgramSetDeviceActionProperties.parse_raw(
    json.dumps(
        {
            "version": ["1"],
            "actionType": "braket.ir.openqasm.program_set",
            "maximumExecutables": 100,
            "maximumTotalShots": 200000,
        }
    )
)

ACTION_PROPERTIES_NO_ADJOINT = OpenQASMDeviceActionProperties.parse_raw(
    json.dumps(
        {
            "actionType": "braket.ir.openqasm.program",
            "version": ["1"],
            "supportedOperations": ["rx", "ry", "h", "cy", "cnot", "unitary"],
            "supportedResultTypes": [
                {
                    "name": "StateVector",
                    "observables": ["x", "y", "z"],
                    "minShots": 0,
                    "maxShots": 0,
                },
            ],
        }
    )
)

ACTION_PROPERTIES_DM_DEVICE = OpenQASMDeviceActionProperties.parse_raw(
    json.dumps(
        {
            "actionType": "braket.ir.openqasm.program",
            "version": ["1"],
            "supportedOperations": ["rx", "ry", "h", "cy", "cnot", "unitary"],
            "supportedResultTypes": [
                {
                    "name": "StateVector",
                    "observables": ["x", "y", "z"],
                    "minShots": 0,
                    "maxShots": 0,
                },
            ],
            "supportedPragmas": [
                "braket_noise_bit_flip",
                "braket_noise_depolarizing",
                "braket_noise_kraus",
                "braket_noise_pauli_channel",
                "braket_noise_generalized_amplitude_damping",
                "braket_noise_amplitude_damping",
                "braket_noise_phase_flip",
                "braket_noise_phase_damping",
                "braket_noise_two_qubit_dephasing",
                "braket_noise_two_qubit_depolarizing",
                "braket_unitary_matrix",
                "braket_result_type_sample",
                "braket_result_type_expectation",
                "braket_result_type_variance",
                "braket_result_type_probability",
                "braket_result_type_density_matrix",
            ],
        }
    )
)

ACTION_PROPERTIES_NATIVE = OpenQASMDeviceActionProperties.parse_raw(
    json.dumps(
        {
            "actionType": "braket.ir.openqasm.program",
            "version": ["1"],
            "supportedOperations": ["rx", "ry", "h", "cy", "cnot", "unitary"],
            "supportedResultTypes": [
                {
                    "name": "StateVector",
                    "observables": None,
                    "minShots": 0,
                    "maxShots": 0,
                },
                {
                    "name": "AdjointGradient",
                    "observables": ["x", "y", "z", "h", "i"],
                    "minShots": 0,
                    "maxShots": 0,
                },
            ],
            "supportedPragmas": ["verbatim"],
        }
    )
)

GATE_MODEL_RESULT = GateModelTaskResult(
    **{
        "measurements": [[0, 0], [0, 0], [0, 0], [1, 1]],
        "measuredQubits": [0, 1],
        "taskMetadata": {
            "braketSchemaHeader": {
                "name": "braket.task_result.task_metadata",
                "version": "1",
            },
            "id": "task_arn",
            "shots": 100,
            "deviceId": "default",
        },
        "additionalMetadata": {
            "action": {
                "braketSchemaHeader": {
                    "name": "braket.ir.openqasm.program",
                    "version": "1",
                },
                "source": "qubit[2] q; cnot q[0], q[1]; measure q;",
            },
        },
    }
)

RESULT = GateModelQuantumTaskResult.from_string(
    json.dumps(
        {
            "braketSchemaHeader": {
                "name": "braket.task_result.gate_model_task_result",
                "version": "1",
            },
            "measurements": [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]],
            "resultTypes": [
                {"type": {"targets": [0], "type": "probability"}, "value": [0.5, 0.5]},
                {
                    "type": {
                        "observable": ["x"],
                        "targets": [1],
                        "type": "expectation",
                    },
                    "value": 0.0,
                },
                {
                    "type": {"observable": ["y"], "targets": [2], "type": "variance"},
                    "value": 0.1,
                },
                {
                    "type": {"observable": ["z"], "targets": [3], "type": "sample"},
                    "value": [1, -1, 1, 1],
                },
            ],
            "measuredQubits": [0, 1, 2, 3],
            "taskMetadata": {
                "braketSchemaHeader": {
                    "name": "braket.task_result.task_metadata",
                    "version": "1",
                },
                "id": "task_arn",
                "shots": 0,
                "deviceId": "default",
            },
            "additionalMetadata": {
                "action": {
                    "braketSchemaHeader": {
                        "name": "braket.ir.openqasm.program",
                        "version": "1",
                    },
                    "source": "qubit[2] q; cnot q[0], q[1]; measure q;",
                },
            },
        }
    )
)

OQC_PULSE_PROPERTIES_WITH_PORTS = json.dumps(
    {
        "braketSchemaHeader": {
            "name": "braket.device_schema.pulse.pulse_device_action_properties",
            "version": "1",
        },
        "supportedQhpTemplateWaveforms": {},
        "ports": {
            "channel_15": {
                "portId": "channel_15",
                "direction": "tx",
                "portType": "port_type_1",
                "dt": 5e-10,
            },
            "channel_13": {
                "portId": "channel_13",
                "direction": "tx",
                "portType": "port_type_1",
                "dt": 5e-10,
            },
        },
        "supportedFunctions": {},
        "frames": {
            "q0_drive": {
                "frameId": "q0_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q1_drive": {
                "frameId": "q1_drive",
                "portId": "channel_13",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
        },
        "supportsLocalPulseElements": False,
        "supportsDynamicFrames": True,
        "supportsNonNativeGatesWithPulses": True,
        "validationParameters": {
            "MAX_SCALE": 1.0,
            "MAX_AMPLITUDE": 1.0,
            "PERMITTED_FREQUENCY_DIFFERENCE": 1.0,
            "MIN_PULSE_LENGTH": 8e-09,
            "MAX_PULSE_LENGTH": 0.00012,
        },
    }
)

OQC_PULSE_PROPERTIES_ALL_FRAMES = json.dumps(
    {
        "braketSchemaHeader": {
            "name": "braket.device_schema.pulse.pulse_device_action_properties",
            "version": "1",
        },
        "supportedQhpTemplateWaveforms": {},
        "ports": {},
        "supportedFunctions": {},
        "frames": {
            "q0_drive": {
                "frameId": "q0_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q0_second_state": {
                "frameId": "q0_second_state",
                "portId": "channel_15",
                "frequency": 4.5e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q1_drive": {
                "frameId": "q1_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q1_second_state": {
                "frameId": "q1_second_state",
                "portId": "channel_15",
                "frequency": 4.5e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q2_drive": {
                "frameId": "q2_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q2_second_state": {
                "frameId": "q2_second_state",
                "portId": "channel_15",
                "frequency": 4.5e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q3_drive": {
                "frameId": "q3_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q3_second_state": {
                "frameId": "q3_second_state",
                "portId": "channel_15",
                "frequency": 4.5e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q4_drive": {
                "frameId": "q4_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q4_second_state": {
                "frameId": "q4_second_state",
                "portId": "channel_15",
                "frequency": 4.5e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q5_drive": {
                "frameId": "q5_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q5_second_state": {
                "frameId": "q5_second_state",
                "portId": "channel_15",
                "frequency": 4.5e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q6_drive": {
                "frameId": "q6_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q6_second_state": {
                "frameId": "q6_second_state",
                "portId": "channel_15",
                "frequency": 4.5e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q7_drive": {
                "frameId": "q7_drive",
                "portId": "channel_15",
                "frequency": 4.6e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
            "q7_second_state": {
                "frameId": "q7_second_state",
                "portId": "channel_15",
                "frequency": 4.5e9,
                "centerFrequency": 4360000000.0,
                "phase": 0.0,
                "associatedGate": None,
                "qubitMappings": [0],
                "qhpSpecificProperties": None,
            },
        },
        "supportsLocalPulseElements": False,
        "supportsDynamicFrames": True,
        "supportsNonNativeGatesWithPulses": True,
        "validationParameters": {
            "MAX_SCALE": 1.0,
            "MAX_AMPLITUDE": 1.0,
            "PERMITTED_FREQUENCY_DIFFERENCE": 1.0,
            "MIN_PULSE_LENGTH": 8e-09,
            "MAX_PULSE_LENGTH": 0.00012,
        },
    }
)

OQC_PARADIGM_PROPERTIES = json.dumps(
    {
        "braketSchemaHeader": {
            "name": "braket.device_schema.gate_model_qpu_paradigm_properties",
            "version": "1",
        },
        "connectivity": {
            "fullyConnected": False,
            "connectivityGraph": {
                "0": ["1", "7"],
                "1": ["2"],
                "2": ["3"],
                "4": ["3", "5"],
                "6": ["5"],
                "7": ["6"],
            },
        },
        "qubitCount": 8,
        "nativeGateSet": ["ecr", "i", "rz", "v", "x"],
    }
)
