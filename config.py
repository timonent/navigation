import os
import ctypes
import platform

#----------------------------------------------------------------------------
# Hyperparameters.

class Hyperparameters:
    batch_size       = 32
    replay_mem_size  = 2048
    lr               = 2.5e-4
    gamma            = 0.99
    tau              = 0.001
    epsilon          = 1.0
    epsilon_min      = 0.01
    epsilon_decay    = 0.9999
    episodes         = 1000
    t_max            = 1000
    fc_size          = 1024
    nof_filters      = [128, 128, 256]
    frame_stack_size = 4
    update_interval  = 4

hyperparameters = Hyperparameters()

#----------------------------------------------------------------------------
# Paths.

model_dir = 'models'
plot_dir = 'plots'

os_name = platform.system()
if os_name == 'Linux':
    bit_mode = '_64' if ctypes.sizeof(ctypes.c_voidp) == 8 else ''
    env_file = os.path.join('VisualBanana_Linux', f'Banana.x86{bit_mode}')
    assert os.path.isfile(env_file)
elif os_name == 'Darwin':
    env_file = 'VisualBanana.app'
    assert os.path.isdir(env_file)
elif os_name == 'Windows':
    bit_mode = '_64' if ctypes.sizeof(ctypes.c_voidp) == 8 else ''
    env_file = os.path.join(f'VisualBanana_Windows_x86{bit_mode}', 'Banana.exe')
    assert os.path.isfile(env_file)
else:
    print(f'Error during configuration: OS \'{os_name}\' not supported!')
    exit(1)

#----------------------------------------------------------------------------
