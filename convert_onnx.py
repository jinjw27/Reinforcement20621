import torch
import torch.onnx
from stable_baselines3 import PPO

# 안정적인 Export용 래퍼
class PolicyForExport(torch.nn.Module):
    def __init__(self, sb3_policy):
        super().__init__()
        # SB3 정책 내부 모듈 구성
        self.features_extractor = sb3_policy.features_extractor
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net

    def forward(self, x):
        # SB3 ActorCriticPolicy forward 분해
        features = self.features_extractor(x)
        latent_pi, _ = self.mlp_extractor(features)
        action_logits = self.action_net(latent_pi)
        return action_logits

def convert_model(model_path, onnx_path):
    # 모델 불러오기
    model = PPO.load(model_path)
    policy = PolicyForExport(model.policy)
    policy.eval()

    # 더미 입력 (배치 1, obs_dim = 환경 관측 공간 크기)
    dummy_input = torch.randn(1, model.observation_space.shape[0], dtype=torch.float32)

    # ONNX Export
    torch.onnx.export(
        policy,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],         # ✅ 웹 코드와 동일
        output_names=['action'],             # ✅ 웹에서 참조할 출력 이름
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    print(f"ONNX model exported to {onnx_path}")

if __name__ == "__main__":
    convert_model("final_model.zip", "model.onnx")
