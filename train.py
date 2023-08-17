
from torch import nn
import torch
import numpy as np
from tqdm import trange


from pydmd import ParametricDMD,DMD
from ezyrb import POD,RBF



dmd = DMD(svd_rank=20)
rom = POD(rank=20)
interpolator = RBF()

pdmd_monolithic = ParametricDMD(dmd, rom, interpolator)

TRAIN_SIZE=80
TEST_SIZE=20
numtimes=102

outputs=np.load("outputs.npy")
outputs=np.swapaxes(outputs,1,2)
print(outputs.shape)
parameters=np.load("inputs.npy")[:,0,1].reshape(-1,1)


parameters_train=parameters[:TRAIN_SIZE]
parameters_test=parameters[TRAIN_SIZE:]

outputs_train=outputs[:TRAIN_SIZE]
outputs_test=outputs[TRAIN_SIZE:]
outputs_pred_train=outputs_train[:,:,-1]
outputs_train=outputs_train[:,:,:-1]
outputs_pred_test=outputs_test[:,:,-1]
outputs_test=outputs_test[:,:,:-1]


pdmd_monolithic.fit(
    outputs_train, parameters_train
)

pdmd_monolithic.parameters=parameters_test
result = pdmd_monolithic.reconstructed_data
print(np.max(np.abs((result-outputs_test)/outputs_test)))
time_step=1/101



pdmd_monolithic.dmd_time["t0"] = (
    pdmd_monolithic.original_time["tend"] 
)
pdmd_monolithic.dmd_time["tend"] = (
    pdmd_monolithic.original_time["tend"] + 101
)

print(
    f"ParametricDMD will compute {len(pdmd_monolithic.dmd_timesteps)} timesteps:",
    pdmd_monolithic.dmd_timesteps * time_step,
)

result = pdmd_monolithic.reconstructed_data
print(np.max(np.abs((result[:,:,101]-outputs_pred_test)/outputs_pred_test)))

