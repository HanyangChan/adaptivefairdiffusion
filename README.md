# Step-aware Adaptive Fairness Guidance in Diffusion Models

본 프로젝트는 Diffusion 기반 이미지 생성 모델에서 발생하는 성별(Gender) 및 인종(Ethnicity) 등 민감한 속성에 대한 편향(Bias)을 완화하면서도, 생성 이미지의 품질(Quality) 손실을 최소화하기 위한 **동적 공정성 스케줄링(Dynamic Fairness Scheduling)** 기법을 제안합니다.

## 📖 1. 개요 (Overview)
기존의 **Fair Diffusion** 등의 방법론들은 생성 과정(Denoising Steps) 내내 일정한 강도(Constant Weight)로 Fairness Guidance를 주입했습니다. 이는 공정성을 달성할 수는 있으나, 원본 프롬프트가 의도했던 형태적 구조나 시맨틱(Semantic)을 파괴하여 전반적인 생성 품질(CLIP Score, FID)을 심각하게 떨어뜨리는 딜레마(Trade-off)를 안고 있었습니다.

본 연구에서는 Diffusion의 Denoising Step마다 각기 다른 역할을 수행한다는 점에 착안하여, **시간 축에 따라 Fairness Guidance 강도를 동적으로 스케줄링(Adaptive Scheduling)** 하고, 원본 시맨틱을 보존하는 **Dynamic Prompting**을 결합하여 완벽한 Quality-Fairness Pareto Frontier를 달성했습니다.

---

## 🚀 2. 발전 과정 (Development Process)

본 연구는 다음과 같은 세 번의 진화 과정을 거쳐 완성되었습니다.

### Phase 1: 초기 가설 (Forward Scheduling)
- **가설**: "Diffusion의 초반 Step은 형태와 구조를 잡고, 후반 Step은 디테일과 텍스처를 잡는다. 따라서 초반에는 개입을 최소화하여 품질을 살리고, 후반에 강하게 개입하여 공정성을 맞추자." (초기 약함 → 후기 강함)
- **결과**: 실패. 사람의 성별(Gender)은 디테일이 아니라 '초기 구조(Structure)' 단계에서 뼈대와 함께 결정된다는 것을 발견했습니다. 후기에 아무리 강력한 Guidance를 주어도 이미 남성/여성으로 고정된 구조를 뒤집을 수 없었습니다.

### Phase 2: 가설 수정 (Reverse Scheduling)
- **가설**: "성별 구조가 고정되기 전인 **초기 Step에 매우 강력한 Fairness Guidance를 주입하고, 후반부로 갈수록 강도를 줄여서(Reverse Cosine)** 자연스러운 텍스처(Quality)를 살리자." (초기 강함 → 후기 약함)
- **결과**: 대성공. 남성으로 극단적 편향(~95%)되어 있던 의사(Doctor) 프롬프트와 여성 편향이었던 간호사(Nurse) 프롬프트를 거의 50:50으로 완벽하게 뒤집는 데 성공했습니다. 하지만 강한 초기 개입으로 인해 여전히 약간의 화질(CLIP Score) 하락이 존재했습니다.

### Phase 3: 최종 최적화 (Semantic Preserving Dynamic Prompting)
- **가설**: "단순히 `diverse ethnicities, equally mixed genders` 라는 프롬프트 방향으로 모델을 강제로 밀어붙이면 원본인 '의사'라는 시맨틱이 훼손된다. **원본 프롬프트의 의미를 보존하면서 속성만 바꾸자.**"
- **결과**: `fairness_prompt`를 단순히 편향 완화 단어들만 넣는 것이 아니라, `f"{base_prompt}, diverse ethnicities, equally mixed genders"` 로 동적으로 결합시켰습니다. 그 결과, Quality 손실을 완벽하게 방어하며 최상위권의 품질을 달성했습니다.

---

## ⚙️ 3. 방법론 (Methodology)

### 3-way Classifier-Free Guidance (CFG)
매 Step $t$마다 모델은 3번의 추론을 진행합니다.
1. `noise_pred_uncond`: Unconditional (Null)
2. `noise_pred_text`: Base Prompt (예: "A portrait of a doctor")
3. `noise_pred_fair`: Dynamic Fairness Prompt (예: "A portrait of a doctor, diverse ethnicities...")

결종적인 노이즈 예측값은 다음과 같이 계산됩니다:
$$ \epsilon_{pred} = \epsilon_{uncond} + s_{cfg} \cdot (\epsilon_{text} - \epsilon_{uncond}) + \mathbf{w_{fair}(t)} \cdot (\epsilon_{fair} - \epsilon_{text}) $$

### Reverse Cosine Scheduler
시간에 따른 동적 가중치 $\mathbf{w_{fair}(t)}$는 초기($t \rightarrow T$)에 가장 큰 값(`max_w`)을 가지고, 후기($t \rightarrow 0$)로 갈수록 부드럽게 감소하도록 설계되었습니다.
```python
def reverse_cosine(step, total_steps, max_w, min_w=0.0):
    progress = step / max(1, total_steps - 1)
    cosine_val = 0.5 * (1 - math.cos(progress * math.pi))
    return max_w - (max_w - min_w) * cosine_val
```

---

## 📊 4. 실험 결과 (Results)

제안하는 `Reverse Cosine + Dynamic Prompting` 기법을 기존의 베이스라인들(No Guidance, Fair Diffusion(Constant), FairGen(Mid 25%))과 다수의 시드(Multi-seed) 상에서 비교 평가했습니다.

### 최적화 결과 (Quality-Fairness Trade-off)
![Optimized Quality-Fairness Trade-off](tradeoff_plot_optimized.png)

위 그래프에서 우상단에 위치할수록 좋은 방법론입니다. 
- 파란색(Constant)이나 녹색(FairGen) 등의 기존 기법들은 Fairness를 위해 Quality를 포기하거나, Quality를 위해 Fairness를 포기하는 전형적인 Trade-off를 보여줍니다.
- 하지만 저희가 제안한 **Ours (Reverse + Dynamic Prompt) 기법(빨간색 별 ★)**은 기존 방식들을 아득히 뛰어넘어, 기존 품질 범위를 유지(CLIP Score 26~28 방어)하면서 동시에 완벽에 가까운 공정성(Fairness Score ~1.0)을 달성하는 **압도적인 Pareto Frontier 우위**를 입증했습니다.

---

## 💻 5. 사용 방법 (How to Run)

**1. 환경 설정**
```bash
pip install -r requirements.txt
```

**2. 전체 Ablation Study 재현**
```bash
python experiment_runner.py
```
- 결과 이미지와 JSON 지표는 `results_optimized/` 에 저장됩니다.

**3. Trade-off 커브 도식화**
```bash
python plot_tradeoff.py
```
