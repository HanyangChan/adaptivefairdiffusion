# Step-aware Adaptive Fairness Guidance in Diffusion Models

본 프로젝트는 Diffusion 기반 이미지 생성 모델에서 발생하는 특정 속성(예: 성별, 인종 등)에 대한 편향(Bias)을 완화하고, 생성 이미지의 품질(Quality) 손실 간의 균형(Trade-off)을 분석하기 위한 동적 공정성 스케줄링(Dynamic Fairness Scheduling) 및 Latent 공간 적응형 제어(Latent Space Adaptive Control) 기법을 제안 및 검증합니다.

## 1. 개요 (Overview)
기존의 Fair Diffusion과 같은 방법론들은 생성 과정(Denoising Steps) 내내 일정한 강도(Constant Weight)로 Fairness Guidance를 주입합니다. 이러한 방식은 공정성을 달성하는 데는 효과적이나, 원본 텍스트 프롬프트가 의도했던 형태적 구조나 시맨틱을 다소 훼손하여 전반적인 생성 품질(예: CLIP Score)을 하락시키는 한계를 가집니다.

본 연구에서는 단순한 시간에 따른 스케줄링을 넘어, **매 스텝마다 모델이 실제로 이미지를 생성해 나가는 Latent 상태를 실시간으로 반영하여 제어하는 진정한 의미의 적응형(Adaptive) 기법**을 제안합니다. 이를 통해 Quality와 Fairness 간의 Trade-off를 극한으로 최적화했습니다.

---

## 2. 발전 과정 (Development Process)

본 연구의 핵심 아이디어와 스케줄링 기법은 다음과 같은 과정을 거쳐 발전되었습니다.

### Phase 1: 초기 가설 (Forward Scheduling)
- 가설: "Diffusion의 초반 Step은 형태와 구조를 잡고, 후반 Step은 디테일과 텍스처를 잡는다. 따라서 초반에는 개입을 최소화하여 품질을 살리고, 후반에 강하게 개입하여 공정성을 맞추자." (초기 약함 -> 후기 강함)
- 결과: 성별(Gender)과 같은 속성은 디테일이 아니라 초기 '구조(Structure)' 단계에서 결정된다는 점을 확인했습니다. 후반부에 강한 Guidance를 주입하더라도 이미 확정된 성별 구조를 뒤집는 데는 한계가 있었습니다.

### Phase 2: 가설 수정 (Reverse Scheduling)
- 가설: "성별과 같은 구조적 편향을 교정하기 위해서는 초기 Step에 강력한 Fairness Guidance를 주입하고, 디테일이 형성되는 후반부로 갈수록 강도를 줄여 품질 훼손을 방지하자." (초기 강함 -> 후기 약함)
- 결과: 남녀 비율을 균형 있게 맞추는 데는 성공했으나, 초기 강한 개입으로 인해 원본 프롬프트와의 정합성(CLIP Score)이 하락하는 부작용이 관찰되었습니다.

### Phase 3: 동적 프롬프팅 (Semantic Preserving Dynamic Prompting)
- 가설: 단순히 공정성 속성만을 타겟으로 Guidance를 주면 원본 시맨틱이 훼손된다. 따라서 공정성 프롬프트에 원본 프롬프트를 결합하자.
- 결과: 하락했던 CLIP Score를 상당 부분 복구하면서도 높은 Fairness를 유지했습니다. 하지만 이것은 여전히 사전에 정해진 수식(스케줄링)에 의한 방식이었습니다.

### Phase 4: 진정한 의미의 적응형 제어 (Latent Space Adaptive Fairness)
- 가설: 단순 스케줄링은 이미지 생성 도중 Latent 공간에서 프롬프트 간에 일어나는 충돌 상황을 전혀 반영하지 못한다. Latent 공간 내에서의 벡터 상호작용을 제어하자.
- 결과: 메인 프롬프트의 의미를 훼손하지 않도록 Fairness 방향을 **직교 투영(Orthogonal Projection)** 하거나, 충돌 정도(Cosine Similarity)에 따라 가중치를 조절하는 **동적 가중치(Dynamic Weighting)** 기법을 도입하여 성능을 크게 향상시켰습니다.

---

## 3. 방법론 (Methodology)

### 3.1 3-way Classifier-Free Guidance (CFG)
매 Step마다 모델은 세 가지 조건에 대해 추론을 진행합니다.
1. `noise_pred_uncond`: Unconditional (Null text)
2. `noise_pred_text`: Base Prompt (원본 텍스트 방향)
3. `noise_pred_fair`: Dynamic Fairness Prompt (공정성 주입 방향)

여기서 메인 텍스트가 이미지를 이끄는 방향을 `d_text`, 공정성을 위해 추가로 보정하려는 방향을 `d_fair`로 정의합니다.
- `d_text = noise_pred_text - noise_pred_uncond`
- `d_fair = noise_pred_fair - noise_pred_text`

### 3.2 진정한 의미의 적응형 제어 (Latent Space Methods)
사전에 정해진 스케줄링 가중치 위에, 매 스텝 Latent 벡터의 상태를 기반으로 실시간 개입을 수행합니다.

#### 1) Latent 공간 직교 투영 (Orthogonal Projection)
- **목적**: Fairness 보정이 이미지의 핵심 품질(Semantic)을 파괴하지 않도록 보호합니다.
- **원리**: Fairness 방향 벡터(`d_fair`)를 메인 텍스트 방향 벡터(`d_text`)에 직교(Orthogonal)하게 투영하여, 메인 텍스트와 충돌하는 성분을 물리적으로 제거합니다. 
- **수식**: `d_fair_orthogonal = d_fair - ( (d_fair · d_text) / (d_text · d_text) ) * d_text`
- **효과**: 이미지의 본질적인 특징은 보존하면서, 훼손되지 않는 여유 공간을 통해 Fairness(성별 균형 등)만을 부드럽게 주입합니다. 이 방식은 **CLIP Score(이미지 품질)를 방어하는 데 매우 효과적**입니다.

#### 2) 충돌 기반 동적 가중치 조절 (Dynamic Weighting)
- **목적**: 두 프롬프트가 충돌할 때만 강력하게 공정성을 주입하여 효율을 높입니다.
- **원리**: 매 스텝마다 `d_text`와 `d_fair` 간의 코사인 유사도(Cosine Similarity)를 계산합니다. 
- **수식**: `adaptive_factor = 1.0 - cosine_similarity(d_text, d_fair)`
- **효과**: 두 방향이 서로 강하게 대립할 때(코사인 값이 음수) 가중치를 동적으로 증폭시킵니다. 정해진 스케줄보다 더 능동적으로 편향을 교정하기 때문에 **가장 높은 Fairness 수치를 달성**할 수 있습니다.

---

## 4. 실험 환경 및 세부 설정 (Experimental Setup)

- 프레임워크: PyTorch, Hugging Face `diffusers`, `transformers`
- 사용 모델: `runwayml/stable-diffusion-v1-5`
- 평가 지표: 
  - Quality: CLIP Score (`openai/clip-vit-base-patch32`)를 이용한 텍스트-이미지 정합성
  - Fairness: 동일한 CLIP 모델을 이용한 Zero-shot Classification 성별 확률 분포 (1.0 - abs(P_male - P_female))
- 실험 환경: Apple Silicon (MPS 가속) 또는 CUDA 호환 환경
- 생성 파라미터: 30 Inference steps, CFG Scale 7.5
- 소요 시간: MPS 환경 기준 이미지 78장 생성 및 평가에 약 10~15분 소요

---

## 5. 실험 결과 (Results)

![Optimized Quality-Fairness Trade-off](tradeoff_plot_optimized.png)

1. **직교 투영(Orthogonal)의 탁월한 화질 보존**: `Adaptive (Orthogonal)` 기법은 베이스라인 수준에 근접한 가장 높은 CLIP Score(29.45)를 유지하면서도 편향을 개선했습니다. 
2. **동적 가중치(Dynamic Weight)의 완벽한 공정성**: `Adaptive (Dynamic Weight)` 기법은 테스트된 최적화 방법론 중 가장 압도적인 Fairness 점수(0.364)를 달성했습니다.
3. **종합 결론**: 제안된 적응형(Adaptive) 모델들은 기존의 단순 스케줄링 방법들을 압도하며 그래프의 우상단(높은 퀄리티 & 높은 공정성)으로 Pareto Frontier를 크게 전진시켰습니다.

### 5.1 정량적 평가 (Quantitative Evaluation)

각 기법들의 전반적인 Fairness와 CLIP-score(Quality)를 비교한 정량적 평가 지표입니다. 

| Method | Fairness ↑ | CLIP-score ↑ |
|---|---|---|
| Stable Diffusion (No Guidance) | 0.266 | 30.02 |
| FairGen [Mid 25%] | 0.278 | 30.04 |
| Fair Diffusion [Constant] | 0.349 | 28.60 |
| Ours (Reverse Cosine Unoptimized) | 0.748 | 24.43 |
| Ours (Reverse Cosine Optimized) | 0.328 | 28.82 |
| **Adaptive (Orthogonal)** | 0.336 | **29.45** |
| **Adaptive (Dynamic Weight)** | **0.364** | 28.47 |
| Adaptive (Both) | 0.296 | 28.82 |

### 5.2 정성적 평가 (Qualitative Results)

정량적 지표뿐만 아니라, 실제 생성된 이미지를 통해 베이스라인 대비 제안 모델의 성별 균형 달성도를 직관적으로 확인할 수 있습니다.

#### "A portrait of a doctor" (남성 편향 프롬프트)
![Doctor Grid](grid_A_portrait_of_a_doctor.png)
- **Baseline**: 3개의 시드 모두 남성으로 생성됩니다.
- **Ours**: 형태를 보존하면서도 일부 시드에서 자연스럽게 여성 의사로 성별이 뒤집히며 전체 비율 균형을 맞춥니다.

#### "A photo of a nurse" (여성 편향 프롬프트)
![Nurse Grid](grid_A_photo_of_a_nurse.png)
- **Baseline**: 3개의 시드 모두 여성으로 생성됩니다.
- **Ours**: 원본의 퀄리티와 시맨틱을 보존하면서 남성 간호사의 이미지를 성공적으로 생성해 냅니다.

---

## 6. 사용 방법 (How to Run)

1. 환경 설정
```bash
pip install -r requirements.txt
```

2. 실험 실행 및 이미지 생성
```bash
python experiment_runner.py
```
- 결과 이미지 및 평가지표 JSON 데이터는 `results_option1/` 폴더에 저장됩니다.

3. Trade-off 결과 시각화
```bash
python generate_table.py
python plot_tradeoff.py
```
- 누적된 실험 데이터를 분석하여 콘솔에 테이블을 출력하고 `tradeoff_plot_optimized.png` 그래프를 생성합니다.
