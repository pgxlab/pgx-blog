+++
title = "18시간 뒤의 침묵 — DeepGEMM JIT 커널 크래시 추적기"
date = 2026-09-04T16:35:26+09:00
draft = false
tags = ["DGX Spark", "GB10", "vLLM", "GLM-5.3", "DeepGEMM", "트러블슈팅", "장애분석"]
summary = "정상 종료 코드로 죽은 추론 서버, TileLang 대체 경로로의 우회, 그리고 '조용한 모니터링 장애'까지 — 4노드 클러스터 장애 하나를 끝까지 따라간 기록"
+++

> 운영 문서에서 IP·호스트명·계정·자격증명을 제거한 버전. 노드는 node1~node4로 표기.

4노드 DGX Spark(GB10) 클러스터에서 GLM-5.3-Flash NVFP4를 TP=4로 서빙 중이다. 18시간 정상 가동하던 서버가 조용히 죽었다. 원인을 찾고 우회하고 검증한 과정을 정리한다.

## 1. 증상 — 정상 종료 코드로 죽은 서버

발견은 단순한 상태 점검이었다. 헤드 노드(rank 0)의 컨테이너가 없다.

```
vllm_glm53   Exited (0) 3 hours ago
```

**종료 코드가 0이다.** 크래시라면 보통 137(SIGKILL)이나 1이 찍힌다. 0은 "정상 종료"다.

더 이상한 건 워커 쪽이었다. node2~4의 컨테이너는 `Up 21 hours`로 멀쩡히 살아 있었다.

```
node1 (head)   Exited (0)
node2 (rank1)  Up 21 hours
node3 (rank2)  Up 21 hours
node4 (rank3)  Up 21 hours
```

살아있는 껍데기였다. 워커 로그를 보니 헤드가 죽고 약 2시간 반 뒤 NCCL 하트비트 모니터가 뒤늦게 상황을 알아챘다.

```
[rank1] Failed to check the "should dump" flag on TCPStore,
        (maybe TCPStore server has shut down too early), with error: Broken pipe
```

이 사실 하나가 나중에 "자동 재기동" 설계 전체를 좌우한다. 헤드만 되살려도 워커의 프로세스 그룹은 이미 깨져 있어 재결합이 안 된다.

## 2. 진짜 오류

컨테이너 로그를 뒤로 감으니 종료 직전에 이게 있었다.

```
RuntimeError: CUDA driver error
  (deepgemm-src/csrc/apis/../jit_kernels/impls/../../jit/handle.hpp:154):
  800 (CUDA_ERROR_NOT_PERMITTED, operation not permitted)
```

종료 코드 0의 정체는 이거였다. 엔진이 치명적 오류를 **잡아서** API 서버를 graceful shutdown 시킨 것이다. 오류는 로그에만 남고, 프로세스는 예의 바르게 0을 반환하며 종료했다.

여기서 첫 번째 실용적 교훈이 나온다. **Docker `--restart on-failure`는 이 크래시를 못 잡는다.** "failure"가 아니기 때문이다.

전체 호출 경로:

```
GLM-5.3 mhc(hyper-connection) pre-norm 레이어
  → torch.ops.vllm.mhc_pre_tilelang    (kernels/mhc/tilelang.py:195)
  → tf32_hc_prenorm_gemm               (utils/deep_gemm.py:669)
  → DeepGEMM 런타임 JIT 컴파일
  → CUDA driver 800: NOT_PERMITTED
```

## 3. 배제 작업

`CUDA_ERROR_NOT_PERMITTED`는 흔한 오류가 아니다. 후보를 하나씩 지웠다.

| 가설 | 확인 방법 | 결과 |
|---|---|---|
| 드라이버 자동 업데이트 | `apt history.log`, `dpkg.log` 해당 시간대 | 변경 없음 |
| unattended-upgrades 개입 | 해당 로그 | 무동작 |
| GPU 메모리 부족 | `dmesg` | `NV_ERR_NO_MEMORY`가 있긴 한데 **18시간 전** — 컨테이너 기동 직후 KV캐시 할당 구간의 알려진 현상, 무관 |
| 컨테이너 권한 문제 | `docker inspect` | privileged=false, GPU device-request 정상, 18시간 잘 돌았음 |
| GPU 하드웨어 이상 | `nvidia-smi` | 현재 정상, idle |

특히 dmesg의 OOM 로그는 함정이었다. 시각을 맞춰보지 않았다면 그럴듯한 범인이 될 뻔했다. 크래시와 18시간 차이가 난다.

**컨테이너 로그는 UTC, 호스트 `date`는 KST**라는 것도 이 과정에서 걸려 넘어진 지점이다. 9시간 차이를 놓치면 인과관계가 통째로 뒤집힌다.

## 4. "처음 실행된 경로"였을까?

가장 그럴듯한 가설은 "18시간 동안 한 번도 안 타던 코드 경로에 특정 요청이 처음 진입했다"였다. 이거라면 요청 형태에 종속된 문제이므로 재기동해도 같은 요청이 오면 또 죽는다.

로그를 확인하니 — **아니었다.**

같은 함수 안에서 `tf32_hc_prenorm_gemm` 바로 다음에 실행되는 `mhc_pre_big_fuse_with_norm_tilelang`이 크래시 16시간 전에 JIT 컴파일을 마친 기록이 있었다. 이 경로는 이미 여러 번 통과했다.

남은 가설은 둘이다.

1. shape 조합별로 갈리는 DeepGEMM per-shape JIT 변종의 컴파일/로드 실패
2. 일시적인 드라이버 상태 문제

**어느 쪽인지는 아직 모른다.** 배경으로 짚이는 건 있다. DeepGEMM은 Hopper(SM90)와 데이터센터 Blackwell(SM100) 위주로 검증된 라이브러리인데, GB10은 SM121a다. 검증 범위 밖의 아키텍처에서 드라이버 레벨 커널 속성 요청이 거부됐을 수 있다 — 어디까지나 추정이다.

## 5. 조치 — 소스에서 분기점 찾기

원인을 확정 못 했으니 원인 제거는 불가능하다. 대신 **그 코드 경로 자체를 안 타게** 할 수 있는지 봤다. 컨테이너 이미지 안의 vLLM 소스를 직접 열었다.

```python
# kernels/mhc/tilelang.py:167
use_deep_gemm = is_deep_gemm_supported()
...
if use_deep_gemm:
    tf32_hc_prenorm_gemm(...)      # ← 크래시 지점
else:
    _tilelang_hc_prenorm_gemm(...)  # ← TileLang 자체 구현
```

대체 구현이 이미 있었다. 그리고 분기 조건은:

```python
def is_deep_gemm_supported() -> bool:
    return envs.VLLM_USE_DEEP_GEMM and has_deep_gemm() and is_supported_arch
```

**환경변수 하나로 끌 수 있다.** 런처에 한 줄 추가했다.

```
-e VLLM_USE_DEEP_GEMM=0
```

부작용 검토도 했다. DeepGEMM의 다른 주 용도는 FP8 MoE/linear인데, 이 구성은 NVFP4(compressed-tensors) 가중치에 MoE 백엔드가 marlin이라 해당 없다. 실제로 크래시 이전 부팅 로그에서도 DeepGEMM 관련 라인은 초기화 메시지 2줄이 전부였다.

## 6. 검증

재기동 후 확인한 것들.

**대체 경로가 실제로 돌았나** — 이전 부팅 로그엔 없던 커널이 컴파일됐다.

```
TileLang begins to compile kernel `hc_prenorm_gemm_tilelang`
TileLang begins to compile kernel `hc_prenorm_gemm_block_m_tilelang`
```

`DeepGEMM E8M0 enabled` 라인은 사라졌다. 우회 성공.

**부팅 시간** — 13분 14초. 이전 10분 38초 대비 +2분 36초. TileLang 대체 커널의 최초 JIT 컴파일 비용이고, 캐시가 컨테이너와 함께 사라지므로 **재기동마다 반복된다.**

**성능** — 단일 스트림 측정.

| 유형 | 기존 기록 | 우회 후 |
|---|---|---|
| 한국어 산문 256tok | 21~30 tok/s | 32.2 (3회 평균) |
| 영문 코드 400tok | 38~48 tok/s | 54.8 / 66.1 |
| 숫자 나열 300tok | 70 tok/s | 98.9 |
| 투기 디코딩 수락률 | 31.8% | 35.0% |

숫자만 보면 빨라진 것 같지만, **그렇게 해석하면 안 된다.** 기존 기록 자체가 동일 프롬프트에서 38~48 tok/s의 측정 변동을 남기고 있고, 이번 측정은 재기동 직후 완전 유휴 상태라는 유리한 조건이었다. 말할 수 있는 건 "**저하가 없다**"까지다.

## 7. 곁다리로 잡힌 조용한 장애

크래시와 별개로, 상태 점검 중에 모니터링 대시보드의 메모리 게이지가 4개가 아니라 2개만 떠 있는 걸 발견했다.

익스포터 컨테이너는 4노드 전부 정상 가동 중이었다. Prometheus 타깃을 직접 조회하니:

```
node1 node-exporter up
node2 node-exporter up
node3 node-exporter down   ← no route to host
node4 node-exporter down   ← no route to host
```

며칠 전 네트워크 장비 재배선 과정에서 node3·4의 관리망 DHCP 주소가 바뀌었는데, Prometheus의 스크레이프 설정은 옛 주소를 붙들고 있었다. 익스포터도 살아있고 대시보드도 그려지니 **아무것도 에러를 내지 않는다.** 게이지 개수가 줄어들 뿐이다.

조치는 스크레이프 타깃 전체를 DHCP의 영향을 안 받는 오버레이 VPN 주소로 옮기는 것이었다. 근본적으로는 **모니터링 대상 주소가 변할 수 있는 값이면 안 된다**는 이야기다.

이런 부류의 장애가 제일 위험하다. 크래시는 시끄럽지만, 이건 "잘 돌아가는 것처럼 보이는데 절반만 보고 있는" 상태다.

## 8. 자동 재기동은 왜 간단하지 않은가

"크래시 감지해서 자동 재기동"은 얼핏 `--restart always` 한 줄처럼 보인다. 아니다.

- 종료 코드가 0이라 `on-failure`는 발동하지 않는다
- `always`로 헤드만 되살려도 **워커의 NCCL 프로세스 그룹이 이미 깨져 있다.** 1절의 그 로그다. 4노드 동시 재기동이 필수다
- 부팅에 10~13분이 걸린다. 크래시가 요청 형태에 종속된 것이었다면, 클라이언트가 같은 요청을 재시도 → 재기동 → 또 크래시의 무한 루프가 된다. **재기동 횟수 제한이 감지 로직보다 중요하다**
- 콜드 부팅은 1시간을 넘을 수도 있어서 "부팅 유예 15분" 같은 고정값을 두면 watchdog이 부팅 중인 서버를 계속 죽인다. 상태기계가 필요하다
- `/health`는 NCCL이 hang인 상태에서도 200을 반환할 수 있다. 실제 추론 요청 프로브를 병행해야 한다

그래서 watchdog은 설계만 해두고 우선 **알림**부터 붙였다. Grafana 내장 alerting으로 두 개.

| 규칙 | 조건 | 지속 |
|---|---|---|
| 서빙 다운 | `up{job="vllm"} < 1`, NoData도 알림 | 5분 |
| 익스포터 타깃 다운 | `count(up{job=~"node-exporter\|gpu-exporter"} == 0) > 0` | 10분 |

두 번째 규칙이 7절의 조용한 장애를 잡는 그물이다. 각 규칙의 annotation에는 대응 절차(4노드 전체 재기동 필요, 로그 확인 명령, 문서 절 번호)를 넣어뒀다. 새벽에 알림을 받는 사람이 알림 본문만 보고 움직일 수 있어야 한다.

여담으로, Grafana 13에서는 contact point "테스트" 버튼용 구 API가 제거됐고 신 API는 빈 body를 거부한다. `vector(1) > 0`으로 즉시 firing하는 임시 규칙을 만들어 실제 전달까지 확인하고 지우는 편이 낫다 — 규칙→정책→전달 전 구간을 다 태우니 오히려 더 확실한 검증이다.

## 9. 정리

- 종료 코드 0으로 죽는 크래시가 있다. 재시작 정책만 믿으면 안 된다
- 분산 추론에서 헤드가 죽으면 워커는 **살아있는 채로 쓸모없어진다**. 복구는 항상 전체 단위
- 원인을 확정 못 해도 조치는 가능하다. 소스에서 분기점을 찾으면 문제 경로를 통째로 우회할 수 있는 경우가 있다
- 벤치 숫자가 좋아졌다고 원인 제거의 근거로 삼지 말 것. 측정 조건이 달랐을 뿐일 수 있다
- 크래시보다 무서운 건 조용한 부분 장애다. 모니터링 자체를 감시하는 규칙이 필요하다
- 로그 타임스탬프의 타임존을 먼저 확인할 것

DeepGEMM 우회가 실제로 근본 대책이었는지는 며칠 무사고로 돌아봐야 안다. 재발하면 원인 가설이 "shape별 JIT"에서 "드라이버 상태"로 좁혀진다. 그때 다시 쓰겠다.
