+++
title = "정정: DGX Spark의 FP4는 네이티브다 — 그리고 그걸 켜도 빨라지지 않았다"
date = 2026-09-05T16:20:00+09:00
draft = false
tags = ["DGX Spark", "GB10", "vLLM", "NVFP4", "marlin", "CUTLASS", "FlashInfer", "벤치마크", "정정"]
summary = "앞 글에서 GB10의 NVFP4를 '에뮬레이션'이라고 쓴 것은 틀렸다. 하드웨어도 빌드도 FP4를 지원하고, marlin은 레시피의 선택이었다. 그래서 FP4 텐서코어 경로로 바꿔 재봤더니 전 구간 같거나 느렸다. 왜 그런지, 그리고 flashinfer 경로는 왜 부팅조차 못 했는지."
+++

> 운영 문서에서 IP·호스트명·계정·자격증명을 제거한 버전. 노드는 node1~node4로 표기.

## 틀린 문장

[앞 글](../2026-09-05-concurrency-vs-rtx-pro-6000/)의 하드웨어 비교표에 이렇게 썼다.

> NVFP4: RTX PRO 6000은 네이티브 텐서코어, DGX Spark GB10은 에뮬레이션(marlin 커널)

근거는 vLLM 부팅 로그의 이 경고였다.

```
Your GPU does not have native support for FP4 computation but FP4 quantization
is being used. Weight-only FP4 compression will be used leveraging the Marlin kernel.
```

게시 직후 지적을 받았다. "스파크에서 NVFP4가 네이티브 지원이 아닌 게 확실해?" 확실하지 않았다. 확인했다.

## 확인한 것

**하드웨어.** GB10은 Blackwell이고 컴퓨트 능력 12.1이다(`nvidia-smi --query-gpu=compute_cap`). 5세대 텐서코어에 FP4 연산이 있다. NVIDIA가 DGX Spark 사양에 적는 "1 PFLOP FP4"가 바로 그것이다.

**빌드.** 우리가 쓰는 vLLM 컨테이너(DGX Spark용 빌드)도 SM121용 CUTLASS NVFP4 GEMM을 갖고 있다. 컨테이너 안에서:

```python
from vllm._custom_ops import cutlass_scaled_mm_supports_fp4
cutlass_scaled_mm_supports_fp4(121)   # True
```

**그럼 왜 marlin인가.** 레시피의 실행 스크립트가 `--moe-backend marlin`을 명시했기 때문이다. 부팅 로그를 다시 보면 후보가 다 나열돼 있다.

```
Using 'MARLIN' NvFp4 MoE backend out of potential backends:
['FLASHINFER_TRTLLM', 'FLASHINFER_CUTEDSL', 'FLASHINFER_CUTLASS',
 'VLLM_CUTLASS', 'MARLIN', 'HUMMING', 'EMULATION'].
```

**그 경고 문구는.** 소스를 보니 `marlin_utils_fp4.py`의 marlin MoE 준비 함수 안에 `logger.warning_once(...)`로 박혀 있다. marlin 경로에 들어가면 하드웨어와 무관하게 무조건 찍힌다. 하드웨어 판정 결과가 아니라 marlin 경로의 안내문이다. 나는 안내문을 판정으로 읽었다.

정리하면: FP4 텐서코어는 있고, 커널도 있고, 설정이 marlin을 고른 것이다. "에뮬레이션"은 하드웨어 차이가 아니라 우리 설정이었다.

## 그렇다면 켜면 빨라지는가

재기동 두 번으로 확인했다. 나머지 설정은 앞 글과 동일(`--max-num-seqs 64`, KV 16GiB, DeepGEMM off, enforce-eager). 벤치는 앞 글과 같은 스크립트(고유 코드 생성 프롬프트, 300/600 토큰, temperature 0, 벽시계 합산)에 프리필 측정(무작위 단어 11K·23K 토큰, 1토큰 출력, 프리픽스 캐시 무효)을 추가했다.

| MoE 백엔드 | 부팅 | c=1 | c=6 | c=32 | c=64 (300tok) | c=64 (600tok) | 프리필 11K / 23K |
|---|---|---|---|---|---|---|---|
| `auto` → FLASHINFER_CUTLASS | **실패** | – | – | – | – | – | – |
| `cutlass` (VLLM_CUTLASS, FP4 텐서코어) | 정상 | 65 | 92 | 220 | 256 | 285 | 1,668 / 1,684 tok/s |
| `marlin` (weight-only, 레시피 기본) | 정상 | 69 | 110~111 | 243 | 273~280 | 305 | 1,891 / 1,853 tok/s |

단위는 tok/s. marlin은 앞 글 측정과 이번 복귀 후 재측정을 함께 적었다(c=6 111→110, c=64 280→273 — 재현성은 이 정도).

### 1차: auto — 부팅 실패

`--moe-backend`를 지우면 vLLM이 우선순위대로 고른다. TRTLLM과 CuteDSL은 이 구성을 지원하지 않는다고 거절됐고 FLASHINFER_CUTLASS가 선택됐다. 가중치 로드까지 정상, 프로파일링 단계에서 워커가 죽었다. flashinfer가 SM121용 fused MoE 모듈을 **런타임에 nvcc로 JIT 컴파일**하는데, 그 빌드가 이렇게 끝났다.

```
tensorrt_llm/deep_gemm/jit_utils.cuh:21:10: fatal error: nvrtc.h: No such file or directory
ninja: build stopped: subcommand failed.
```

`nvrtc.h`는 컨테이너 안에 있다. pip 패키지 `nvidia-cuda-nvrtc`가 `.../nvidia/cu13/include/nvrtc.h`에 깔아 놨다. 그런데 nvcc가 보는 `/usr/local/cuda/include/`에는 없다. 하드웨어 문제도 커널 문제도 아닌 컨테이너 패키징 문제다. 심볼릭 링크 하나로 풀릴 가능성이 높지만, 이번엔 시도하지 않았다. 부팅 한 번이 15분이고 오늘 이미 네 번째였다.

### 2차: cutlass — 정상, 그러나 느림

`--moe-backend cutlass`(vLLM 자체 CUTLASS, 미리 빌드된 커널)는 정상 부팅했고 64 동시 요청까지 오류 없이 처리했다. 결과는 표대로다. 디코딩 -8~-17%, 프리필 -10% 안팎. 1회 측정이라 ±10%는 잡음으로 보되, **빨라진 구간이 하나도 없다.**

## 왜 FP4 텐서코어가 이득이 없나

확실한 것부터. 이 클러스터에서 디코딩은 메모리 대역폭에 묶여 있다(273 GB/s). marlin은 FP4로 압축된 가중치를 그대로 읽어 bf16으로 계산하는 weight-only 경로다. 즉 **가중치 대역폭 절감은 이미 누리고 있다.** FP4 텐서코어 경로가 추가로 주는 것은 연산 속도인데, 디코딩은 연산이 병목이 아니다. 대신 활성값까지 FP4로 양자화하는 단계가 붙는다. 배치가 작은 디코딩에서는 그 비용을 상쇄할 연산 이득이 없다. 여기까지는 근거가 있는 해석이다.

프리필에서도 이득이 없는 이유는 확실치 않다. 프리필은 연산 비중이 크므로 FP4 텐서코어가 유리해야 한다. 그런데도 marlin이 앞섰다. 4노드 RoCE 텐서 병렬 통신이 프리필 시간을 지배해서 GEMM 속도 차이가 묻히는 것일 수 있고, vLLM CUTLASS 경로가 SM121에서 아직 튜닝되지 않았을 수도 있다. 둘 다 추정이다. 앞 글에서 "남은 격차는 대역폭·eager·노드 간 통신"이라고 했는데, 이번 결과는 그중 노드 간 통신의 비중이 생각보다 클 가능성을 시사한다.

## 결론

- GB10의 NVFP4는 네이티브다. 앞 글의 표는 고쳤다.
- 레시피가 marlin을 고른 것은 이 시스템에서 타당하다. marlin으로 복귀했다.
- flashinfer 경로는 `nvrtc.h` 인클루드 경로 문제로 부팅 불가. 컨테이너 쪽 수정 과제로 남긴다.
- 원래 주장의 결론(RTX PRO 6000과의 남은 20배 격차는 설정이 아니라 하드웨어·런타임)은 유지된다. 하지만 그 근거 중 하나가 틀렸던 것은 틀렸던 것이다. 로그의 경고문은 소스를 열어 보기 전까지는 사실이 아니다.

## 운영 메모

오늘 재기동 네 번(동시성 실험 1, 백엔드 실험 2, 복귀 1). 계획 재기동을 워치독으로 하면 재기동 이력에 남아 자동 재기동 예산(6시간 2회)을 먹는다. 실제로 2차 실험 중 워치독이 "6시간 내 2회 초과"로 포기 상태에 들어갔다. 실험 후 이력을 정리해 예산을 복원했고, 다음부터는 실험 전 워치독을 해제하고 실험 후 이력을 정리하는 것을 절차에 넣는다.
