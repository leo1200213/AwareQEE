from numpy import isin
import torch
from conv import MinMaxQuantConv2d
from linear import MinMaxQuantLinear, PTQSLQuantLinear
from matmul import MinMaxQuantMatMul, PTQSLQuantMatMul
import torch.nn.functional as F
from tqdm import tqdm
from models.swin_highway.modeling_qswin import SwinLayer

def swin_block_forward_hook(module, input, output):
    print(f"swin_block_forward_hook called for module {module}")
    if not hasattr(module, 'raw_input'):
        module.raw_input = []
    if not hasattr(module, 'raw_out'):
        module.raw_out = []
    module.raw_input.append(input[0].detach())
    if isinstance(output, tuple):
        output = output[0]
    module.raw_out.append(output.detach())
    
class QuantCalibrator():
    """
    Modularization of quant calib.

    Notice: 
    all quant modules has method "calibration_step1" that should only store raw inputs and outputs
    all quant modules has method "calibration_step2" that should only quantize its intervals
    and we assume we could feed in all calibration data in one batch, without backward propagations

    sequential calibration is memory-friendly, while parallel calibration may consume 
    hundreds of GB of memory.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=True,batch_size=1):
        self.net = net
        self.wrapped_modules = wrapped_modules
        self.calib_loader = calib_loader
        self.sequential = sequential
        self.calibrated = False
        self.batch_size = batch_size
    
    def sequential_quant_calib(self):
        """
        A quick implementation of calibration.
        Assume calibration dataset could be fed at once.
        """
        # run calibration
        n_calibration_steps=2
        for step in range(n_calibration_steps):
            print(f"Start calibration step={step+1}")
            for name,module in self.wrapped_modules.items():
                # corner cases for calibrated modules
                if hasattr(module, "calibrated"):
                    if step == 1:
                        module.mode = "raw"
                    elif step == 2:
                        print('step2')
                        module.mode = "quant_forward"
                else:
                    module.mode=f'calibration_step{step+1}'
            with torch.no_grad():
                #i = 0
                for batch in self.calib_loader:
                    inp = batch['pixel_values']
                    target = batch['labels']
                    #print('-------------------------------------inp-------------------------------------\n', inp)
                    #print('-------------------------------------target-------------------------------------\n', target)
                    #print(f"---------------------------type of inp(before)---------------------------{type(inp)}")
                    inp = inp.cuda()
                    #print(f"---------------------------type of (after)---------------------------{type(inp)}")
                    #self.net.to('cuda')
                    self.net(inp, disable_early_exits=True)
                    #print(i)
                        
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("sequential calibration finished")
    
    def parallel_quant_calib(self):
        """
        A quick implementation of parallel quant calib
        Assume calibration dataset could be fed at once, and memory could hold all raw inputs/outs
        """
        # calibration step1: collect raw data
        print(f"Start calibration step=1")
        for name,module in self.wrapped_modules.items():
            # corner cases for calibrated modules
            if hasattr(module, "calibrated"):
                module.mode = "raw"
            else:
                module.mode=f'calibration_step1'
        with torch.no_grad():
            for inp,target in self.calib_loader:
                inp=inp.cuda()
                self.net(inp, disable_early_exits=True)
        # calibration step2: each module run calibration with collected raw data
        for name,module in self.wrapped_modules.items():
            if hasattr(module, "calibrated"):
                continue
            else:
                module.mode=f"calibration_step2"
                with torch.no_grad():
                    if isinstance(module, MinMaxQuantLinear):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantConv2d):
                        module.forward(module.raw_input.cuda())
                    elif isinstance(module, MinMaxQuantMatMul):
                        module.forward(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                    torch.cuda.empty_cache()
                
        # finish calibration
        for name,module in self.wrapped_modules.items():
            module.mode='quant_forward'
        torch.cuda.empty_cache() # memory footprint cleanup
        print("calibration finished")
    
    def quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")
        if self.sequential:
            self.sequential_quant_calib()
        else:
            self.parallel_quant_calib()
        self.calibrated = True

    def batching_quant_calib(self):
        calib_layers = []
        for name, module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start calibration")
        for name, module in self.wrapped_modules.items():
            module.mode = 'raw'
        # Assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)
            module.mode = 'calibration_step2'
            # Add forward hooks to current modules
            hooks = []
            if isinstance(module, SwinLayer):
                hooks.append(module.register_forward_hook(swin_block_forward_hook))
            elif isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            elif isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            elif isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            
            # Feed in calibration data and store the data
            for batch in self.calib_loader:
                inp = batch['pixel_values']
                target = batch['labels']
                for batch_st in range(0, len(inp), self.batch_size):
                    self.net.zero_grad()
                    inp_ = inp[batch_st:batch_st+self.batch_size].to(next(self.net.parameters()).device)
                    self.net(inp_, disable_early_exits=True)
                del inp, target, batch
                torch.cuda.empty_cache()
            
            # Concatenate collected inputs and outputs if they exist
            if hasattr(module, 'raw_input') and module.raw_input:
                if isinstance(module, SwinLayer):
                    module.raw_input = torch.cat(module.raw_input, dim=0)
                    module.raw_out = torch.cat(module.raw_out, dim=0)
                    # Run calibration step2 for SwinLayer
                    with torch.no_grad():
                        module.calibration_step2(module.raw_input.to(next(self.net.parameters()).device))
                elif isinstance(module, MinMaxQuantLinear):
                    module.raw_input = torch.cat(module.raw_input, dim=0)
                    module.raw_out = torch.cat(module.raw_out, dim=0)
                    with torch.no_grad():
                        module.calibration_step2(module.raw_input.to(next(self.net.parameters()).device))
                elif isinstance(module, MinMaxQuantConv2d):
                    module.raw_input = torch.cat(module.raw_input, dim=0)
                    module.raw_out = torch.cat(module.raw_out, dim=0)
                    with torch.no_grad():
                        module.calibration_step2(module.raw_input.to(next(self.net.parameters()).device))
                elif isinstance(module, MinMaxQuantMatMul):
                    module.raw_input = [
                        torch.cat(module.raw_input[0], dim=0),
                        torch.cat(module.raw_input[1], dim=0)
                    ]
                    module.raw_out = torch.cat(module.raw_out, dim=0)
                    with torch.no_grad():
                        module.calibration_step2(
                            module.raw_input[0].to(next(self.net.parameters()).device),
                            module.raw_input[1].to(next(self.net.parameters()).device)
                        )
                # Clean up temporary data
                module.raw_input = None
                module.raw_out = None
            else:
                print(f"No data collected for module {name}")

            # Remove hooks for the current module
            for hook in hooks:
                hook.remove()

            # Finish calibration for the current module
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"
            module.mode = 'quant_forward'
        # Finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("calibration finished")


def grad_hook(module, grad_input, grad_output):
    if module.raw_grad is None:
        module.raw_grad = []
    module.raw_grad.append(grad_output[0].detach())   # that's a tuple!

def linear_forward_hook(module, input, output):
    #print(f"linear_forward_hook called for module {module}")
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].detach())
    module.raw_out.append(output.detach())
"""
def linear_forward_hook(module, input, output):
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].detach())
    module.raw_out.append(output.detach())
"""
def conv2d_forward_hook(module, input, output):
    #print(f"conv_forward_hook called for module {module}")
    if module.raw_input is None:
        module.raw_input = []
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input.append(input[0].detach())
    module.raw_out.append(output.detach())

def matmul_forward_hook(module, input, output):
    #print(f"matmul_forward_hook called for module {module}")
    if module.raw_input is None:
        module.raw_input = [[],[]]
    if module.raw_out is None:
        module.raw_out = []
    module.raw_input[0].append(input[0].detach())
    module.raw_input[1].append(input[1].detach())
    module.raw_out.append(output.detach())

class HessianQuantCalibrator(QuantCalibrator):
    """
    Modularization of hessian_quant_calib

    Hessian metric needs gradients of layer outputs to weigh the loss,
    which calls for back propagation in calibration, both sequentially
    and parallelly. Despite the complexity of bp, hessian quant calibrator
    is compatible with other non-gradient quantization metrics.
    """
    def __init__(self, net, wrapped_modules, calib_loader, sequential=False, batch_size=1):
        super().__init__(net, wrapped_modules, calib_loader, sequential=sequential)
        self.batch_size = batch_size

    def quant_calib(self):
        """
        An implementation of original hessian calibration.
        """

        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start hessian calibration")

        # get raw_pred as target distribution 
        with torch.no_grad():
            for batch  in self.calib_loader:
                inp = batch['pixel_values']
                raw_pred = self.net(inp.cuda(),disable_early_exits=True)
                if isinstance(raw_pred, tuple):
                    raw_pred = raw_pred[0]  # Extract logits from the tuple
                raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()
            torch.cuda.empty_cache()

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Brecq")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric") and module.metric == "hessian":
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the data
            for batch in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp = batch['pixel_values']
                    target = batch['labels']
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    pred = self.net(inp_, disable_early_exits=True)
                    if isinstance(pred, tuple):
                        pred = pred[0]  # Extract logits from the tuple
                    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")
                    loss.backward()
                del inp, target, pred, loss, batch
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric") and module.metric == "hessian":
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2(module.raw_input.cuda())
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2(module.raw_input[0].cuda(), module.raw_input[1].cuda())
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")

    def batching_quant_calib(self):
        calib_layers=[]
        for name,module in self.wrapped_modules.items():
            calib_layers.append(name)
        print(f"prepare parallel calibration for {calib_layers}")

        print("start hessian calibration")

        # get raw_pred as target distribution 
        with torch.no_grad():
            for batch in self.calib_loader:
                inp = batch['pixel_values']
                raw_pred = self.net(inp.cuda(), disable_early_exits=True)
                if isinstance(raw_pred, tuple):
                    raw_pred = raw_pred[0]  # Extract logits from the tuple
                raw_pred_softmax = F.softmax(raw_pred, dim=-1).detach()
            torch.cuda.empty_cache()

        # assume wrapped modules are in order (true for dict in python>=3.5)
        q = tqdm(self.wrapped_modules.items(), desc="Hessian")
        for name, module in q:
            q.set_postfix_str(name)

            # add fp and bp hooks to current modules, which bypass calibration step 1
            # precedent modules are using quant forward
            hooks = []
            if isinstance(module, MinMaxQuantLinear):
                hooks.append(module.register_forward_hook(linear_forward_hook))
            if isinstance(module, MinMaxQuantConv2d):
                hooks.append(module.register_forward_hook(conv2d_forward_hook))
            if isinstance(module, MinMaxQuantMatMul):
                hooks.append(module.register_forward_hook(matmul_forward_hook))
            if hasattr(module, "metric"):
                hooks.append(module.register_backward_hook(grad_hook))
            
            # feed in calibration data, and store the dat
            for batch in self.calib_loader:
                for batch_st in range(0,self.calib_loader.batch_size,self.batch_size):
                    self.net.zero_grad()
                    inp = batch['pixel_values']
                    target = batch['labels']
                    inp_ = inp[batch_st:batch_st+self.batch_size].cuda()
                    pred = self.net(inp_, disable_early_exits=True)
                    if isinstance(pred, tuple):
                        pred = pred[0]  # Extract logits from the tuple
                    loss = F.kl_div(F.log_softmax(pred, dim=-1), raw_pred_softmax[batch_st:batch_st+self.batch_size], reduction="batchmean")
                    loss.backward()
                del inp, target, pred, loss, batch
                torch.cuda.empty_cache()
            
            # replace cached raw_inputs, raw_outs
            if isinstance(module, MinMaxQuantLinear):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantConv2d):
                module.raw_input = torch.cat(module.raw_input, dim=0)
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if isinstance(module, MinMaxQuantMatMul):
                module.raw_input = [torch.cat(_, dim=0) for _ in module.raw_input]
                module.raw_out = torch.cat(module.raw_out, dim=0)
            if hasattr(module, "metric"):
                module.raw_grad = torch.cat(module.raw_grad, dim=0)
            for hook in hooks:
                hook.remove()

            # run calibration step2
            with torch.no_grad():
                if isinstance(module, MinMaxQuantLinear):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantConv2d):
                    module.calibration_step2()
                if isinstance(module, MinMaxQuantMatMul):
                    module.calibration_step2()
                torch.cuda.empty_cache()
            
            # finishing up current module calibration
            if self.sequential:
                module.mode = "quant_forward"
            else:
                module.mode = "raw"

        # finish calibration
        for name, module in self.wrapped_modules.items():
            module.mode = "quant_forward"
        
        print("hessian calibration finished")