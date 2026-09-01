+++
title = "DGX Spark 4노드 TP=4 GLM-5.3-Flash 서빙 — 기술 런북"
date = 2026-09-01T09:30:00+09:00
draft = false
tags = ["DGX Spark", "GB10", "MikroTik", "RoCE", "vLLM", "GLM-5.3", "런북"]
summary = "MikroTik CRS804 브레이크아웃 설정, CX-7 인터페이스 매핑, NCCL/vLLM 파라미터, 실측 성능과 단계별 소요시간"
+++


> 실제 운영 문서에서 IP·호스트명·계정·자격증명·내부 포트맵을 제거한 버전. 노드는 node1~node4, 주소는 플레이스홀더(`<node1-roce-ip>` 등)로 표기.

## 1. 하드웨어 · 네트워크 구성

| 항목 | 내용 |
|---|---|
| 노드 | Lenovo ThinkStation PGX(= NVIDIA DGX Spark, GB10) × 4, 노드당 128GB 통합 메모리, CX-7 200G × 2포트 |
| RoCE 스위치 | MikroTik CRS804-4DDQ-hRM (4× QSFP56-DD 400G), RouterOS 7.20.x |
| 케이블 | 400G QSFP-DD → 2× 200G QSFP56 브레이크아웃 DAC(NADDOD) 4본 → 노드당 2링크, 총 8× 200G |
| 관리망 | 별도 1GbE 스위치 |
| 저장소 | 콜드: Synology 2베이 NAS(1GbE, NFSv4.1) / 핫: node1 NVMe를 RoCE 패브릭 위 NFS로 공유 |

### 1.0 브레이크아웃 케이블 사양
| 항목 | 내용 |
|---|---|
| 제품 | NADDOD Q2Q56-400G-CU 시리즈(사용: 0.5m, 4본) — 400G QSFP-DD → 2× 200G QSFP56 패시브 DAC |
| 전기 | 8×50G PAM4 → 4×50G PAM4 ×2 (200GBASE-CR4), 28~30 AWG, CDR 없음, <0.1W |
| EEPROM | QSFP-DD측 CMIS 5.0, QSFP56측 Identifier 0x11 / "50GBASE-CR, 100GBASE-CR2, or 200GBASE-CR4" |
| 준수·환경 | IEEE 802.3, QSFP-DD/QSFP56 MSA, 0~70°C. 라인업 0.5~3m, 1m ≈ US$138 |
| 실측 | 오토네고로는 50G 협상 → 양단 200G 강제 + RS-FEC 자동 → 8링크 200G |

### 1.1 DGX Spark 쪽 인터페이스 매핑 (중요)
GB10에는 CX-7 netdev가 4개 보이지만 **물리 QSFP 포트는 `f1` 쪽 두 개**다.

| netdev | RDMA dev | 역할 |
|---|---|---|
| `enp1s0f1np1` | `rocep1s0f1` | QSFP 포트 1 → 플레인 A (예: 192.168.A.N/24) |
| `enP2p1s0f1np1` | `roceP2p1s0f1` | QSFP 포트 2 → 플레인 B (예: 192.168.B.N/24) |
| `enp1s0f0np0`, `enP2p1s0f0np0` | — | 미연결(`ethtool -m` 실패, "No cable") |

노드 설정(NetworkManager 예):
```bash
nmcli con add type ethernet ifname enp1s0f1np1 con-name roce0 \
  ipv4.method manual ipv4.addresses 192.168.A.N/24 ipv4.never-default yes \
  ipv6.method disabled 802-3-ethernet.mtu 9000 \
  802-3-ethernet.auto-negotiate no 802-3-ethernet.speed 200000 802-3-ethernet.duplex full
# roce1 / enP2p1s0f1np1 동일
```
추가: `arp_ignore=1`, `arp_announce=2`, `rp_filter=2`(두 서브넷이 같은 L2에 있을 때 ARP flux 방지), `ethtool -A <if> rx on tx on`(802.3x), `vm.swappiness=0`.

### 1.2 CRS804 설정 — 브레이크아웃 링크가 뜨지 않을 때
RouterOS는 QSFP56-DD 포트를 레인별 8개 서브인터페이스 `qsfp56-dd-P-1 … P-8`로 노출한다. 2×200G 분할은 **`-1`(레인 1–4)과 `-5`(레인 5–8)** 만 사용.

증상과 해결 순서(실측):
1. 기본 상태: 링크는 뜨지만 **50Gbps**로 협상 — 서브포트의 `advertising`이 단일 레인 모드(≤50G)만 포함
2. `advertise=200G-baseCR4` 지정 → `advertising` 비워지고 no-link (오토네고로는 CR4 광고 불가)
3. **오토네고 끄고 강제**: 
```
/interface/ethernet/set [find name~"qsfp56-dd-[1-4]-(1|5)\$"] auto-negotiation=no speed=200G-baseCR4 fec-mode=auto rx-flow-control=on tx-flow-control=on l2mtu=9216 mtu=9000
/interface/ethernet/set [find name~"qsfp56-dd-[1-4]-(2|3|4|6|7|8)\$"] disabled=yes
/interface/bridge/add name=br-roce protocol-mode=none mtu=9000
:foreach i in=[/interface/ethernet/find name~"qsfp56-dd-[1-4]-(1|5)\$"] do={/interface/bridge/port/add bridge=br-roce interface=$i hw=yes}
```
   노드 쪽도 `ethtool -s <if> autoneg off speed 200000` → 8링크 모두 200Gbps, RS-FEC(fec91)
4. **주의**: `bridge/port/add ... interface=[find ...]` 한 줄 형식은 조용히 실패했음. `:foreach`로 하나씩 추가하고 `/interface/bridge/port/print`로 반드시 확인
5. 확인: `/interface/ethernet/monitor [find name~"qsfp56-dd-[1-4]-(1|5)\$"] once` → `link-ok 200Gbps`

혼잡 제어: 전용 패브릭이라 PFC 대신 802.3x 글로벌 플로우컨트롤 + MTU 9000만 적용(RouterOS PFC 지원 여부 미확인).

### 1.3 검증 수치
- 점보 ping(8972B, DF) 전 노드 상호 OK
- `ib_write_bw -x 3`(RoCEv2 IPv4 GID) 2플레인 동시: **98 + 98 Gb/s** — 플레인당 ~100G 상한은 GB10의 CX-7 기능당 PCIe 대역 한계로, 다른 사용자 보고와 일치
- `rdma_cm`(-R) 모드는 CM 이벤트 오류로 실패 → GID index 명시 방식 사용

## 2. 저장소 계층
- **콜드**: NAS NFSv4.1 공유(`/mnt/nas-models`). DSM 기본은 NFSv3만 켜져 있어 v4.1 별도 활성화 필요. 1GbE라 ~110MB/s — 원본 보관 전용
- **핫**: node1 NVMe 디렉터리를 RoCE 플레인 IP로 NFS export → 다른 노드 `/mnt/hot-models` 마운트(`nfs4, nconnect=8`). 실측 쓰기 942MB/s, 읽기 1.2GB/s
- 원칙: 인터넷 다운로드는 1노드 1회 → NAS 원본 보관 → 서빙 대상만 핫티어로 복사 → 4노드 공통 경로로 로드
- 보안: export는 노드 IP만 허용하고 `root_squash` 유지 권장

## 3. GLM-5.3-Flash NVFP4 TP=4 서빙
레시피: [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) 를 그대로 따르되 패브릭에 맞게 수정.

| 항목 | 값 |
|---|---|
| 체크포인트 | `RedHatAI/GLM-5.3-Flash-NVFP4` (compressed-tensors, 약 185GB) + `incoai/GLM-5.3-Flash-DFlash2` 드래프터(2.3GB) |
| 이미지 | `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` |
| 랭크 | rank0=node1(head, OpenAI 호환 API), rank1~3=worker, `--distributed-executor-backend mp --nnodes 4` |
| NCCL | `NCCL_NET=IB`, `NCCL_IB_HCA=rocep1s0f1`, `NCCL_IB_GID_INDEX=3`, `NCCL_IB_ROCE_VERSION_NUM=2`, `NCCL_IB_ADDR_RANGE=<플레인A/24>`, `NCCL_SOCKET_IFNAME=GLOO_SOCKET_IFNAME=TP_SOCKET_IFNAME=enp1s0f1np1`, `NCCL_CROSS_NIC=0`, `NCCL_IB_MERGE_NICS=0` (플레인 A 단일 사용) |
| 메모리 | `--gpu-memory-utilization 0.80`, `--kv-cache-memory 24GiB`, `--kv-cache-dtype fp8_e4m3`, `--memory 112g` |
| 기타 | `--max-model-len 1048576`, DFlash2 `num_speculative_tokens 7`, `--moe-backend marlin`, `--enforce-eager`, SM121 indexer 패치 파일 마운트 |

### 3.1 실측 결과
- `GPU KV cache size: 3,895,606 tokens, Maximum concurrency for 1,048,576 tokens per request: 3.72x` (레시피와 동일)
- 부팅: 컨테이너 시작 → `/health` 200 약 **8분** (가중치 로드 216초 @ 핫티어 ~850MB/s)
- 첫 요청(콜드 JIT): 261 tok / 11.1초. 워밍 후 코드 생성: 400 tok / 10.7초 ≈ **37 tok/s** (temperature 0)

### 3.2 기동 절차 (순서 엄수)
1. 전 랭크 정지 `docker rm -f vllm_glm53` — 죽어가는 랭크와 랑데부하면 hang
2. 각 노드 `sync; echo 3 > /proc/sys/vm/drop_caches`, 대용량 다운로드 등 메모리 점유 작업 중지
3. 각 노드에서 레시피의 무조건 플러셔(`flusher-unconditional.sh`)를 root로 백그라운드 실행
4. worker 먼저, 15~20초 간격: rank3 → rank2 → rank1 → rank0(head)
5. `curl <head>:8000/health` 200, 로그의 `GPU KV cache size` 확인

### 3.3 실패 사례
- `Free memory on device cuda:0 (103.12/121.63 GiB) ... less than desired GPU memory utilization (0.85, 103.38 GiB)` → 0.85는 head(다른 서비스 상주)와 다운로드 진행 중이던 노드에서 0.3GiB 차이로 실패. **0.80으로 낮추고 drop_caches** 후 성공. 실패는 1~2분 내 컨테이너 종료로 판별됨
- 한 랭크가 죽으면 head는 `Connection closed by peer`로 종료 → 반드시 전체 teardown

## 4. 모니터링
- Prometheus + Grafana(node1) + 전 노드 `node_exporter`(:9100) / `nvidia_gpu_exporter`(:9835), 대시보드 [RodriMora/dgx-spark-grafana-dashboard](https://github.com/RodriMora/dgx-spark-grafana-dashboard) (쿼리의 `vllm:` → `vllm_` 치환)
- **GB10 특이사항**: `node_exporter`의 **`cpufreq` 컬렉터가 hang** → 스크레이프가 쌓여 503 반환. `--no-collector.cpufreq` 필수
- Grafana는 기존 서비스와 포트 충돌 없는 포트로, 기본 비밀번호는 반드시 변경

## 5. 단계별 실측 소요시간 (총 약 9시간, 그중 약 6시간은 ~100Mbps 회선 다운로드 대기)

| 단계 | 실측 | 비고 |
|---|---|---|
| 노드 NIC 설정 (노드당) | 1~2분 | |
| 스위치 설정 | 약 25분 | 순수 적용 5분 이내, 나머지는 50G 협상·브리지 포트 미적용 트러블슈팅 |
| 링크 검증 | 3분 | |
| 컨테이너 이미지 pull 25~31GB | 1~5.5시간 | 회선 공유 시. GHCR은 `unexpected EOF`로 3회 재시작(레이어 캐시 유지) |
| 가중치 185GB HF 다운로드(1노드) | 약 5시간 | 평균 ~10MB/s |
| 가중치 185GB → 핫티어 rsync | 약 6분 | ~500MB/s |
| 드래프터 2.3GB 3노드 배포 | 1분 미만 | |
| 이미지 31GB fan-out (`docker save \| ssh \| docker load`) | 노드당 약 4분 | ssh 암호화·압축 해제가 병목 |
| GLM TP=4 부팅 | 약 8분 | 레시피 문서의 ~20분(콜드 NFS)보다 짧음 |
| 부팅 실패 판정 | 1~2분 | |
| NAS NFS 공유 + 4노드 마운트 | 약 10분 | |
| 핫티어 구성 | 약 5분 | |
| 모니터링 스택 | 이미지 pull 3~5분/노드 + cpufreq 원인 파악 10분 | |

## 6. 교훈 정리
- GB10 물리 포트는 `f1` 인터페이스. `f0`에 IP를 주고 링크가 안 뜬다고 헤매지 말 것
- MikroTik 400G 브레이크아웃은 서브포트 `-1/-5`에 **오토네고 off + 200G-baseCR4 강제**가 핵심. 오토네고로는 50G에 머문다
- RouterOS에서 `find` 결과를 `bridge/port/add`에 한 번에 넘기면 조용히 실패할 수 있다 — 항상 print로 확인
- 통합 메모리 128GB 노드에서 `gpu-memory-utilization 0.85`는 여유가 0.3GiB 수준. 다른 서비스가 있으면 0.80
- 회선이 느리면 "1노드 다운로드 → 패브릭 배포"가 전부다. 핫티어(NVMe over RoCE NFS) 하나로 부팅 시간이 절반 이하로 줄었다
- 공개 문서에는 IP·계정·비밀번호·포트맵·NFS 옵션을 넣지 말 것 (이 문서가 그 결과물)
