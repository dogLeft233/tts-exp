"""Interaction, not the noisy-low versus clean-standard diagonal."""
import numpy as np
def mixed_input(native,noise30,snr_db,scale):
 native=np.asarray(native,dtype=np.float32)
 if scale not in [1.,.5]:raise ValueError('unplanned scale')
 if snr_db is None:
  if noise30 is not None:raise ValueError('clean arm cannot have noise')
  mixed=native.copy()
 else:
  if snr_db not in [30,90] or noise30 is None:raise ValueError('unplanned noise dose')
  noise30=np.asarray(noise30,dtype=np.float32)
  if noise30.shape!=native.shape:raise ValueError('clock mismatch')
  mixed=native+noise30*np.float32(10**((30-snr_db)/20))
 return mixed*np.float32(scale)
def difference_in_differences(noisy_low,noisy_standard,clean_low,clean_standard):
 return (noisy_low-noisy_standard)-(clean_low-clean_standard)
