"""
Convert Pytorch to ONNX Code

Writer : KHS0616
Last Update : 2022-05-04
"""
import torch

class TestModel(torch.nn.Module):
    """ 테스트용 모델 """
    def __init__(self):
        super(TestModel, self).__init__()
        kernel = 3
        self.head = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel, stride=1, padding=kernel//2, bias=True)
        self.body = torch.nn.Sequential(*[torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel, stride=1, padding=kernel//2, bias=True) for _ in range(1)])
        self.tail = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=kernel, stride=1, padding=kernel//2, bias=True)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return y

def get_input_tensor(batch_size=1, channels=3, height=1080, width=1920, mode="nchw"):
    """ ONNX 변환을 위한 더미 데이터 생성 함수 """
    if mode == "nchw":
        return torch.randn(batch_size, channels, height, width, dtype=torch.float32)

def load_model(model_path):
    model = TestModel()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    return model

def convert(model_path, output_path="OUTPUT.onnx"):
    """ ONNX 변환 함수 """
    input_tensor = get_input_tensor()
    model = load_model(model_path)

    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':{
                2:'input_height',
                3:'input_width'
            },
            'output':{
                2:'output_height',
                3:'output_width'
            }
        }
    )

if __name__ == '__main__':
    convert("")