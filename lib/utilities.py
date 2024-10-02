import torch

def loss_function_Z(Z_tk, Y_tk_1, DeltaBm_tk, Delta_tk):
    return torch.mean(torch.norm(Z_tk - Y_tk_1 * (DeltaBm_tk/Delta_tk), dim=1)**2)

def loss_function_Y(Y_tk, Z_tk, Y_tk1, DeltaBm_tk1, f_tk, Delta_tk):
    return torch.mean(torch.abs(Y_tk -Y_tk1 - Delta_tk * f_tk)**2)

class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, architecture, activation=torch.nn.Tanh):
        super().__init__()

        self.activation = activation
        num_layers = len(architecture)

        if num_layers == 0:
            model = [torch.nn.Linear(in_size, out_size)]
        else:
            model = [torch.nn.Linear(in_size, architecture[0])]      
            model.append(activation())
            
            for i in range(0, num_layers-1):
                model.append(torch.nn.Linear(architecture[i], architecture[i+1]))
                model.append(activation())

            model.append(torch.nn.Linear(architecture[-1], out_size))

        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

def check_cuda_memory():
        # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the device (assuming you want to check the first GPU)
        device = torch.device("cuda:0")
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Get allocated memory
        allocated_memory = torch.cuda.memory_allocated(device)
        
        # Get cached memory
        reserved_memory = torch.cuda.memory_reserved(device)
        
        # Calculate free memory
        free_memory = total_memory - reserved_memory
        
        # Convert bytes to GB
        total_memory_gb = total_memory / (1024 ** 3)
        allocated_memory_gb = allocated_memory / (1024 ** 3)
        reserved_memory_gb = reserved_memory / (1024 ** 3)
        free_memory_gb = free_memory / (1024 ** 3)
        
        print(f"Total GPU memory: {total_memory_gb:.2f} GB")
        print(f"Allocated GPU memory: {allocated_memory_gb:.2f} GB")
        print(f"Reserved GPU memory: {reserved_memory_gb:.2f} GB")
        print(f"Free GPU memory: {free_memory_gb:.2f} GB")
    else:
        print("CUDA is not available.")
