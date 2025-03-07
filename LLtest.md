* 单机8卡（没有做负载均衡）
```txt
root@TENCENT64:/workspace/trmt-deepep# export MASTER_ADDR=127.0.0.1
export MASTER_PORT=61007
export WORLD_SIZE=1
export RANK=0

python tests/test_low_latency.py
Allocating buffer size: 1058.145664 MB ...
[rank 0] Dispatch + combine bandwidth: 4.95 GB/s, avg_t=4450.02 us, min_t=4395.84 us, max_t=4461.95 us
[rank 1] Dispatch + combine bandwidth: 4.95 GB/s, avg_t=4453.39 us, min_t=4449.70 us, max_t=4455.14 us
[rank 2] Dispatch + combine bandwidth: 4.95 GB/s, avg_t=4453.41 us, min_t=4451.49 us, max_t=4454.98 us
[rank 3] Dispatch + combine bandwidth: 4.95 GB/s, avg_t=4450.92 us, min_t=4431.42 us, max_t=4467.07 us
[rank 4] Dispatch + combine bandwidth: 4.95 GB/s, avg_t=4452.00 us, min_t=4442.40 us, max_t=4460.22 us
[rank 5] Dispatch + combine bandwidth: 4.95 GB/s, avg_t=4452.04 us, min_t=4442.66 us, max_t=4466.66 us
[rank 6] Dispatch + combine bandwidth: 4.95 GB/s, avg_t=4451.11 us, min_t=4413.34 us, max_t=4473.28 us
[rank 7] Dispatch + combine bandwidth: 4.95 GB/s, avg_t=4453.68 us, min_t=4450.27 us, max_t=4455.26 us


[rank 0] Dispatch bandwidth: 1.65 GB/s, avg_t=4555.00 us | Combine bandwidth: 376.90 GB/s, avg_t=38.57 us
[rank 1] Dispatch bandwidth: 16.88 GB/s, avg_t=445.03 us | Combine bandwidth: 3.50 GB/s, avg_t=4153.00 us
[rank 2] Dispatch bandwidth: 2.14 GB/s, avg_t=3511.00 us | Combine bandwidth: 15.05 GB/s, avg_t=966.16 us
[rank 3] Dispatch bandwidth: 9.21 GB/s, avg_t=816.03 us | Combine bandwidth: 3.91 GB/s, avg_t=3719.00 us
[rank 4] Dispatch bandwidth: 2.08 GB/s, avg_t=3612.00 us | Combine bandwidth: 16.87 GB/s, avg_t=861.81 us
[rank 5] Dispatch bandwidth: 5.81 GB/s, avg_t=1292.00 us | Combine bandwidth: 4.60 GB/s, avg_t=3157.00 us
[rank 6] Dispatch bandwidth: 2.31 GB/s, avg_t=3249.00 us | Combine bandwidth: 12.47 GB/s, avg_t=1166.00 us
[rank 7] Dispatch bandwidth: 66.32 GB/s, avg_t=113.26 us | Combine bandwidth: 3.35 GB/s, avg_t=4336.00 us


[rank 1] Dispatch send/recv time: 35.14 us | Combine send/recv time: 44.11 us
[rank 7] Dispatch send/recv time: 74.53 us | Combine send/recv time: 44.55 us
[rank 0] Dispatch send/recv time: 74.93 us | Combine send/recv time: 43.63 us
[rank 5] Dispatch send/recv time: 34.69 us | Combine send/recv time: 42.79 us
[rank 6] Dispatch send/recv time: 74.89 us | Combine send/recv time: 83.68 us
[rank 4] Dispatch send/recv time: 35.55 us | Combine send/recv time: 43.61 us
[rank 3] Dispatch send/recv time: 74.37 us | Combine send/recv time: 43.13 us
[rank 2] Dispatch send/recv time: 74.98 us | Combine send/recv time: 163.59 us
```

```txt
python tests/test_low_latency.py
Allocating buffer size: 1058.145664 MB ...
[rank 1] Dispatch + combine bandwidth: 31.49 GB/s, avg_t=700.22 us, min_t=684.16 us, max_t=712.96 us
[rank 5] Dispatch + combine bandwidth: 31.55 GB/s, avg_t=698.80 us, min_t=657.60 us, max_t=712.96 us
[rank 2] Dispatch + combine bandwidth: 31.55 GB/s, avg_t=698.84 us, min_t=680.00 us, max_t=710.53 us
[rank 6] Dispatch + combine bandwidth: 31.52 GB/s, avg_t=699.43 us, min_t=681.34 us, max_t=710.85 us
[rank 0] Dispatch + combine bandwidth: 31.53 GB/s, avg_t=699.25 us, min_t=681.66 us, max_t=715.39 us
[rank 4] Dispatch + combine bandwidth: 31.49 GB/s, avg_t=700.07 us, min_t=690.46 us, max_t=704.29 us
[rank 3] Dispatch + combine bandwidth: 31.53 GB/s, avg_t=699.23 us, min_t=685.63 us, max_t=712.13 us
[rank 7] Dispatch + combine bandwidth: 31.53 GB/s, avg_t=699.27 us, min_t=669.98 us, max_t=714.37 us
[rank 0] Dispatch bandwidth: 29.32 GB/s, avg_t=256.19 us | Combine bandwidth: 33.17 GB/s, avg_t=438.23 us
[rank 7] Dispatch bandwidth: 31.46 GB/s, avg_t=238.74 us | Combine bandwidth: 31.74 GB/s, avg_t=457.98 us
[rank 1] Dispatch bandwidth: 23.89 GB/s, avg_t=314.47 us | Combine bandwidth: 38.57 GB/s, avg_t=376.87 us
[rank 5] Dispatch bandwidth: 27.73 GB/s, avg_t=270.86 us | Combine bandwidth: 34.17 GB/s, avg_t=425.43 us
[rank 6] Dispatch bandwidth: 27.00 GB/s, avg_t=278.23 us | Combine bandwidth: 34.71 GB/s, avg_t=418.76 us
[rank 2] Dispatch bandwidth: 27.54 GB/s, avg_t=272.71 us | Combine bandwidth: 34.51 GB/s, avg_t=421.29 us
[rank 4] Dispatch bandwidth: 32.74 GB/s, avg_t=229.44 us | Combine bandwidth: 31.08 GB/s, avg_t=467.76 us
[rank 3] Dispatch bandwidth: 27.89 GB/s, avg_t=269.34 us | Combine bandwidth: 34.28 GB/s, avg_t=424.07 us
[rank 2] Dispatch send/recv time: 35.05 us | Combine send/recv time: 44.66 us
[rank 4] Dispatch send/recv time: 35.68 us | Combine send/recv time: 44.96 us
[rank 3] Dispatch send/recv time: 34.45 us | Combine send/recv time: 42.89 us
[rank 1] Dispatch send/recv time: 34.83 us | Combine send/recv time: 43.09 us
[rank 6] Dispatch send/recv time: 34.73 us | Combine send/recv time: 44.11 us
[rank 0] Dispatch send/recv time: 34.79 us | Combine send/recv time: 42.79 us
[rank 7] Dispatch send/recv time: 34.58 us | Combine send/recv time: 43.47 us
[rank 5] Dispatch send/recv time: 34.42 us | Combine send/recv time: 42.43 us
```


* 单机8卡（做了负载均衡）
```txt
num_tokens, hidden, num_ranks, num_experts 128 7168 8 144
num_tokens, hidden, num_ranks, num_experts 128 7168 8 144
Allocating buffer size: 1058.145664 MB ...
num_tokens, hidden, num_ranks, num_experts 128 7168 8 144
num_tokens, hidden, num_ranks, num_experts 128 7168 8 144
num_tokens, hidden, num_ranks, num_experts 128 7168 8 144
num_tokens, hidden, num_ranks, num_experts 128 7168 8 144
num_tokens, hidden, num_ranks, num_experts 128 7168 8 144
num_tokens, hidden, num_ranks, num_experts 128 7168 8 144
[rank 3] Dispatch + combine bandwidth: 43.19 GB/s, avg_t=510.51 us, min_t=496.58 us, max_t=535.65 us
[rank 1] Dispatch + combine bandwidth: 43.16 GB/s, avg_t=510.88 us, min_t=490.24 us, max_t=538.37 us
[rank 7] Dispatch + combine bandwidth: 43.23 GB/s, avg_t=509.98 us, min_t=487.68 us, max_t=540.35 us
[rank 4] Dispatch + combine bandwidth: 43.20 GB/s, avg_t=510.38 us, min_t=491.26 us, max_t=552.26 us
[rank 0] Dispatch + combine bandwidth: 43.24 GB/s, avg_t=509.86 us, min_t=494.66 us, max_t=526.91 us
[rank 6] Dispatch + combine bandwidth: 43.17 GB/s, avg_t=510.72 us, min_t=498.43 us, max_t=531.46 us
[rank 5] Dispatch + combine bandwidth: 43.22 GB/s, avg_t=510.13 us, min_t=497.38 us, max_t=533.98 us
[rank 2] Dispatch + combine bandwidth: 43.20 GB/s, avg_t=510.37 us, min_t=499.68 us, max_t=528.13 us
[rank 7] Dispatch bandwidth: 42.32 GB/s, avg_t=177.50 us | Combine bandwidth: 44.52 GB/s, avg_t=326.54 us
[rank 6] Dispatch bandwidth: 43.17 GB/s, avg_t=174.01 us | Combine bandwidth: 43.95 GB/s, avg_t=330.79 us
[rank 5] Dispatch bandwidth: 43.12 GB/s, avg_t=174.22 us | Combine bandwidth: 43.87 GB/s, avg_t=331.36 us
[rank 1] Dispatch bandwidth: 43.04 GB/s, avg_t=174.51 us | Combine bandwidth: 44.10 GB/s, avg_t=329.62 us
[rank 4] Dispatch bandwidth: 42.65 GB/s, avg_t=176.13 us | Combine bandwidth: 44.48 GB/s, avg_t=326.80 us
[rank 0] Dispatch bandwidth: 45.90 GB/s, avg_t=163.65 us | Combine bandwidth: 42.74 GB/s, avg_t=340.12 us
[rank 2] Dispatch bandwidth: 43.45 GB/s, avg_t=172.86 us | Combine bandwidth: 43.99 GB/s, avg_t=330.42 us
[rank 3] Dispatch bandwidth: 44.19 GB/s, avg_t=169.99 us | Combine bandwidth: 43.49 GB/s, avg_t=334.28 us
[rank 5] Dispatch send/recv time: 34.55 us | Combine send/recv time: 42.48 us
[rank 0] Dispatch send/recv time: 34.78 us | Combine send/recv time: 42.72 us
[rank 6] Dispatch send/recv time: 34.78 us | Combine send/recv time: 44.49 us
[rank 3] Dispatch send/recv time: 34.57 us | Combine send/recv time: 42.81 us
[rank 4] Dispatch send/recv time: 35.83 us | Combine send/recv time: 45.02 us
[rank 1] Dispatch send/recv time: 34.53 us | Combine send/recv time: 43.01 us
[rank 2] Dispatch send/recv time: 35.07 us | Combine send/recv time: 44.79 us
[rank 7] Dispatch send/recv time: 34.68 us | Combine send/recv time: 43.49 us
```

* 两机16卡（没有做负载均衡，还要用本机IBGDA）
```txt
Allocating buffer size: 1058.145536 MB ...
[rank 0] Dispatch + combine bandwidth: 33.29 GB/s, avg_t=662.39 us, min_t=584.42 us, max_t=780.45 us
[rank 1] Dispatch + combine bandwidth: 33.27 GB/s, avg_t=662.67 us, min_t=570.72 us, max_t=785.50 us
[rank 2] Dispatch + combine bandwidth: 33.21 GB/s, avg_t=663.98 us, min_t=583.58 us, max_t=813.25 us
[rank 3] Dispatch + combine bandwidth: 33.28 GB/s, avg_t=662.48 us, min_t=601.31 us, max_t=774.43 us
[rank 4] Dispatch + combine bandwidth: 33.21 GB/s, avg_t=663.85 us, min_t=609.47 us, max_t=783.52 us
[rank 5] Dispatch + combine bandwidth: 33.22 GB/s, avg_t=663.76 us, min_t=620.03 us, max_t=775.55 us
[rank 6] Dispatch + combine bandwidth: 33.16 GB/s, avg_t=664.99 us, min_t=618.50 us, max_t=767.97 us
[rank 7] Dispatch + combine bandwidth: 33.10 GB/s, avg_t=666.05 us, min_t=632.54 us, max_t=799.10 us


[rank 0] Dispatch bandwidth: 11.71 GB/s, avg_t=641.62 us | Combine bandwidth: 23.67 GB/s, avg_t=614.14 us
[rank 1] Dispatch bandwidth: 11.58 GB/s, avg_t=648.45 us | Combine bandwidth: 24.06 GB/s, avg_t=604.19 us
[rank 2] Dispatch bandwidth: 11.68 GB/s, avg_t=643.36 us | Combine bandwidth: 23.89 GB/s, avg_t=608.36 us
[rank 3] Dispatch bandwidth: 11.55 GB/s, avg_t=650.43 us | Combine bandwidth: 23.42 GB/s, avg_t=620.69 us
[rank 4] Dispatch bandwidth: 11.77 GB/s, avg_t=638.13 us | Combine bandwidth: 23.00 GB/s, avg_t=631.94 us
[rank 5] Dispatch bandwidth: 11.69 GB/s, avg_t=642.41 us | Combine bandwidth: 23.39 GB/s, avg_t=621.46 us
[rank 6] Dispatch bandwidth: 11.89 GB/s, avg_t=631.94 us | Combine bandwidth: 22.90 GB/s, avg_t=634.87 us
[rank 7] Dispatch bandwidth: 12.32 GB/s, avg_t=609.86 us | Combine bandwidth: 22.78 GB/s, avg_t=638.00 us


[rank 0] Dispatch send/recv time: 34.92 us | Combine send/recv time: 42.56 us
[rank 1] Dispatch send/recv time: 34.61 us | Combine send/recv time: 44.02 us
[rank 2] Dispatch send/recv time: 35.23 us | Combine send/recv time: 45.39 us
[rank 3] Dispatch send/recv time: 34.81 us | Combine send/recv time: 44.74 us
[rank 4] Dispatch send/recv time: 35.12 us | Combine send/recv time: 44.52 us
[rank 5] Dispatch send/recv time: 35.65 us | Combine send/recv time: 45.86 us
[rank 6] Dispatch send/recv time: 35.19 us | Combine send/recv time: 44.39 us
[rank 7] Dispatch send/recv time: 34.75 us | Combine send/recv time: 43.49 us




Allocating buffer size: 1058.145536 MB ...

[rank 8] Dispatch + combine bandwidth: 33.25 GB/s, avg_t=663.17 us, min_t=613.06 us, max_t=775.94 us
[rank 9] Dispatch + combine bandwidth: 33.26 GB/s, avg_t=662.94 us, min_t=585.09 us, max_t=764.90 us
[rank 10] Dispatch + combine bandwidth: 33.21 GB/s, avg_t=663.87 us, min_t=609.73 us, max_t=808.06 us
[rank 11] Dispatch + combine bandwidth: 33.20 GB/s, avg_t=664.06 us, min_t=615.01 us, max_t=771.87 us
[rank 12] Dispatch + combine bandwidth: 33.28 GB/s, avg_t=662.45 us, min_t=588.93 us, max_t=777.31 us
[rank 13] Dispatch + combine bandwidth: 33.20 GB/s, avg_t=664.09 us, min_t=617.50 us, max_t=783.01 us
[rank 14] Dispatch + combine bandwidth: 33.23 GB/s, avg_t=663.59 us, min_t=585.95 us, max_t=796.38 us
[rank 15] Dispatch + combine bandwidth: 33.21 GB/s, avg_t=663.84 us, min_t=619.17 us, max_t=770.05 us


[rank 8] Dispatch bandwidth: 12.22 GB/s, avg_t=614.89 us | Combine bandwidth: 22.69 GB/s, avg_t=640.65 us
[rank 9] Dispatch bandwidth: 11.59 GB/s, avg_t=648.23 us | Combine bandwidth: 24.15 GB/s, avg_t=601.85 us
[rank 10] Dispatch bandwidth: 12.02 GB/s, avg_t=624.74 us | Combine bandwidth: 23.27 GB/s, avg_t=624.73 us
[rank 11] Dispatch bandwidth: 12.00 GB/s, avg_t=625.78 us | Combine bandwidth: 23.33 GB/s, avg_t=623.08 us
[rank 12] Dispatch bandwidth: 11.79 GB/s, avg_t=636.87 us | Combine bandwidth: 23.53 GB/s, avg_t=617.76 us
[rank 13] Dispatch bandwidth: 11.89 GB/s, avg_t=631.70 us | Combine bandwidth: 23.60 GB/s, avg_t=615.93 us
[rank 14] Dispatch bandwidth: 11.99 GB/s, avg_t=626.40 us | Combine bandwidth: 23.53 GB/s, avg_t=617.67 us
[rank 15] Dispatch bandwidth: 16.82 GB/s, avg_t=446.71 us | Combine bandwidth: 17.81 GB/s, avg_t=816.02 us


[rank 8] Dispatch send/recv time: 35.33 us | Combine send/recv time: 44.52 us
[rank 9] Dispatch send/recv time: 35.25 us | Combine send/recv time: 43.35 us
[rank 10] Dispatch send/recv time: 35.61 us | Combine send/recv time: 43.62 us
[rank 11] Dispatch send/recv time: 35.18 us | Combine send/recv time: 44.15 us
[rank 12] Dispatch send/recv time: 34.27 us | Combine send/recv time: 42.30 us
[rank 13] Dispatch send/recv time: 35.14 us | Combine send/recv time: 43.63 us
[rank 14] Dispatch send/recv time: 35.09 us | Combine send/recv time: 43.17 us
[rank 15] Dispatch send/recv time: 35.03 us | Combine send/recv time: 44.39 us

```

* 两机16卡，使用了负载均衡，在本机使用IBGDA
```txt
Allocating buffer size: 1058.145536 MB ...
[rank 0] Dispatch + combine bandwidth: 40.02 GB/s, avg_t=550.99 us, min_t=530.75 us, max_t=576.90 us
[rank 1] Dispatch + combine bandwidth: 39.99 GB/s, avg_t=551.33 us, min_t=533.25 us, max_t=579.33 us
[rank 2] Dispatch + combine bandwidth: 40.01 GB/s, avg_t=551.03 us, min_t=535.01 us, max_t=577.54 us
[rank 3] Dispatch + combine bandwidth: 39.96 GB/s, avg_t=551.72 us, min_t=529.86 us, max_t=571.30 us
[rank 4] Dispatch + combine bandwidth: 39.98 GB/s, avg_t=551.48 us, min_t=531.10 us, max_t=584.00 us
[rank 5] Dispatch + combine bandwidth: 40.04 GB/s, avg_t=550.70 us, min_t=531.01 us, max_t=574.82 us
[rank 6] Dispatch + combine bandwidth: 39.98 GB/s, avg_t=551.44 us, min_t=528.35 us, max_t=574.69 us
[rank 7] Dispatch + combine bandwidth: 40.00 GB/s, avg_t=551.24 us, min_t=537.82 us, max_t=574.08 us
[rank 8] Dispatch + combine bandwidth: 40.03 GB/s, avg_t=550.86 us, min_t=530.88 us, max_t=572.64 us
[rank 9] Dispatch + combine bandwidth: 40.03 GB/s, avg_t=550.84 us, min_t=530.69 us, max_t=577.28 us
[rank 10] Dispatch + combine bandwidth: 40.00 GB/s, avg_t=551.14 us, min_t=529.76 us, max_t=574.43 us
[rank 11] Dispatch + combine bandwidth: 39.97 GB/s, avg_t=551.59 us, min_t=527.33 us, max_t=576.19 us
[rank 12] Dispatch + combine bandwidth: 40.01 GB/s, avg_t=551.12 us, min_t=530.78 us, max_t=579.23 us
[rank 13] Dispatch + combine bandwidth: 40.02 GB/s, avg_t=551.00 us, min_t=529.12 us, max_t=571.46 us
[rank 14] Dispatch + combine bandwidth: 40.03 GB/s, avg_t=550.78 us, min_t=532.54 us, max_t=575.26 us
[rank 15] Dispatch + combine bandwidth: 40.04 GB/s, avg_t=550.72 us, min_t=534.88 us, max_t=577.92 us



[rank 0] Dispatch bandwidth: 32.99 GB/s, avg_t=227.69 us | Combine bandwidth: 39.35 GB/s, avg_t=369.47 us
[rank 1] Dispatch bandwidth: 33.64 GB/s, avg_t=223.30 us | Combine bandwidth: 39.11 GB/s, avg_t=371.69 us
[rank 2] Dispatch bandwidth: 34.75 GB/s, avg_t=216.14 us | Combine bandwidth: 38.48 GB/s, avg_t=377.81 us
[rank 3] Dispatch bandwidth: 34.27 GB/s, avg_t=219.19 us | Combine bandwidth: 39.19 GB/s, avg_t=370.91 us
[rank 4] Dispatch bandwidth: 35.13 GB/s, avg_t=213.85 us | Combine bandwidth: 38.91 GB/s, avg_t=373.64 us
[rank 5] Dispatch bandwidth: 35.33 GB/s, avg_t=212.62 us | Combine bandwidth: 39.04 GB/s, avg_t=372.32 us
[rank 6] Dispatch bandwidth: 35.45 GB/s, avg_t=211.90 us | Combine bandwidth: 39.16 GB/s, avg_t=371.20 us
[rank 7] Dispatch bandwidth: 37.18 GB/s, avg_t=202.04 us | Combine bandwidth: 38.34 GB/s, avg_t=379.14 us
[rank 8] Dispatch bandwidth: 33.18 GB/s, avg_t=226.38 us | Combine bandwidth: 39.54 GB/s, avg_t=367.69 us
[rank 9] Dispatch bandwidth: 37.43 GB/s, avg_t=200.70 us | Combine bandwidth: 38.18 GB/s, avg_t=380.74 us
[rank 10] Dispatch bandwidth: 35.44 GB/s, avg_t=211.97 us | Combine bandwidth: 38.44 GB/s, avg_t=378.18 us
[rank 11] Dispatch bandwidth: 37.51 GB/s, avg_t=200.25 us | Combine bandwidth: 37.56 GB/s, avg_t=387.07 us
[rank 12] Dispatch bandwidth: 35.41 GB/s, avg_t=212.16 us | Combine bandwidth: 39.02 GB/s, avg_t=372.53 us
[rank 13] Dispatch bandwidth: 36.53 GB/s, avg_t=205.64 us | Combine bandwidth: 38.52 GB/s, avg_t=377.36 us
[rank 14] Dispatch bandwidth: 37.23 GB/s, avg_t=201.77 us | Combine bandwidth: 38.47 GB/s, avg_t=377.90 us
[rank 15] Dispatch bandwidth: 36.63 GB/s, avg_t=205.07 us | Combine bandwidth: 39.08 GB/s, avg_t=372.01 us


[rank 0] Dispatch send/recv time: 34.90 us | Combine send/recv time: 42.51 us
[rank 1] Dispatch send/recv time: 34.62 us | Combine send/recv time: 44.12 us
[rank 2] Dispatch send/recv time: 35.28 us | Combine send/recv time: 45.46 us
[rank 3] Dispatch send/recv time: 34.82 us | Combine send/recv time: 44.56 us
[rank 4] Dispatch send/recv time: 35.12 us | Combine send/recv time: 44.50 us
[rank 5] Dispatch send/recv time: 35.47 us | Combine send/recv time: 45.76 us
[rank 6] Dispatch send/recv time: 35.19 us | Combine send/recv time: 44.41 us
[rank 7] Dispatch send/recv time: 34.77 us | Combine send/recv time: 43.38 us
[rank 8] Dispatch send/recv time: 35.54 us | Combine send/recv time: 44.69 us
[rank 9] Dispatch send/recv time: 35.22 us | Combine send/recv time: 43.26 us
[rank 10] Dispatch send/recv time: 35.62 us | Combine send/recv time: 43.67 us
[rank 11] Dispatch send/recv time: 35.17 us | Combine send/recv time: 44.02 us
[rank 12] Dispatch send/recv time: 34.18 us | Combine send/recv time: 42.41 us
[rank 13] Dispatch send/recv time: 35.22 us | Combine send/recv time: 43.69 us
[rank 14] Dispatch send/recv time: 35.15 us | Combine send/recv time: 43.00 us
[rank 15] Dispatch send/recv time: 35.05 us | Combine send/recv time: 44.32 us

[rank 0] Dispatch send/recv time: 34.90 us | Combine send/recv time: 42.51 us
[rank 1] Dispatch send/recv time: 34.62 us | Combine send/recv time: 44.12 us
[rank 2] Dispatch send/recv time: 35.28 us | Combine send/recv time: 45.46 us
[rank 3] Dispatch send/recv time: 34.82 us | Combine send/recv time: 44.56 us
[rank 4] Dispatch send/recv time: 35.12 us | Combine send/recv time: 44.50 us
[rank 5] Dispatch send/recv time: 35.47 us | Combine send/recv time: 45.76 us
[rank 6] Dispatch send/recv time: 35.19 us | Combine send/recv time: 44.41 us
[rank 7] Dispatch send/recv time: 34.77 us | Combine send/recv time: 43.38 us
[rank 8] Dispatch send/recv time: 35.54 us | Combine send/recv time: 44.69 us
[rank 9] Dispatch send/recv time: 35.22 us | Combine send/recv time: 43.26 us
[rank 10] Dispatch send/recv time: 35.62 us | Combine send/recv time: 43.67 us
[rank 11] Dispatch send/recv time: 35.17 us | Combine send/recv time: 44.02 us
[rank 12] Dispatch send/recv time: 34.18 us | Combine send/recv time: 42.41 us
[rank 13] Dispatch send/recv time: 35.22 us | Combine send/recv time: 43.69 us
[rank 14] Dispatch send/recv time: 35.15 us | Combine send/recv time: 43.00 us
[rank 15] Dispatch send/recv time: 35.05 us | Combine send/recv time: 44.32 us



```


* 两机16卡，不在本机内使用IBGDA
```txt
root@TENCENT64:/workspace/trmt-deepep# python tests/test_low_latency.py

[rank 0] Dispatch + combine bandwidth: 39.19 GB/s, avg_t=562.60 us, min_t=537.98 us, max_t=608.29 us
[rank 1] Dispatch + combine bandwidth: 39.17 GB/s, avg_t=562.87 us, min_t=539.97 us, max_t=598.18 us
[rank 2] Dispatch + combine bandwidth: 39.17 GB/s, avg_t=562.88 us, min_t=541.54 us, max_t=594.56 us
[rank 3] Dispatch + combine bandwidth: 39.16 GB/s, avg_t=563.02 us, min_t=535.87 us, max_t=594.18 us
[rank 4] Dispatch + combine bandwidth: 39.17 GB/s, avg_t=562.89 us, min_t=543.90 us, max_t=596.51 us
[rank 5] Dispatch + combine bandwidth: 39.20 GB/s, avg_t=562.47 us, min_t=524.99 us, max_t=599.07 us
[rank 6] Dispatch + combine bandwidth: 39.15 GB/s, avg_t=563.12 us, min_t=538.34 us, max_t=600.51 us
[rank 7] Dispatch + combine bandwidth: 39.19 GB/s, avg_t=562.59 us, min_t=545.41 us, max_t=586.88 us
[rank 8] Dispatch + combine bandwidth: 39.20 GB/s, avg_t=562.45 us, min_t=541.76 us, max_t=602.18 us
[rank 9] Dispatch + combine bandwidth: 39.16 GB/s, avg_t=563.04 us, min_t=541.89 us, max_t=586.34 us
[rank 10] Dispatch + combine bandwidth: 39.16 GB/s, avg_t=563.05 us, min_t=537.50 us, max_t=602.27 us
[rank 11] Dispatch + combine bandwidth: 39.16 GB/s, avg_t=563.09 us, min_t=536.54 us, max_t=602.56 us
[rank 12] Dispatch + combine bandwidth: 39.17 GB/s, avg_t=562.85 us, min_t=542.08 us, max_t=597.82 us
[rank 13] Dispatch + combine bandwidth: 39.21 GB/s, avg_t=562.29 us, min_t=545.06 us, max_t=603.39 us
[rank 14] Dispatch + combine bandwidth: 39.18 GB/s, avg_t=562.69 us, min_t=543.17 us, max_t=587.23 us
[rank 15] Dispatch + combine bandwidth: 39.20 GB/s, avg_t=562.50 us, min_t=548.86 us, max_t=590.66 us




[rank 0] Dispatch bandwidth: 30.70 GB/s, avg_t=244.67 us | Combine bandwidth: 37.82 GB/s, avg_t=384.32 us
[rank 1] Dispatch bandwidth: 31.13 GB/s, avg_t=241.29 us | Combine bandwidth: 37.72 GB/s, avg_t=385.38 us
[rank 2] Dispatch bandwidth: 31.52 GB/s, avg_t=238.32 us | Combine bandwidth: 37.68 GB/s, avg_t=385.76 us
[rank 3] Dispatch bandwidth: 36.88 GB/s, avg_t=203.67 us | Combine bandwidth: 34.75 GB/s, avg_t=418.26 us
[rank 4] Dispatch bandwidth: 32.02 GB/s, avg_t=234.58 us | Combine bandwidth: 37.87 GB/s, avg_t=383.88 us
[rank 5] Dispatch bandwidth: 31.78 GB/s, avg_t=236.38 us | Combine bandwidth: 38.46 GB/s, avg_t=377.98 us
[rank 6] Dispatch bandwidth: 32.77 GB/s, avg_t=229.19 us | Combine bandwidth: 37.87 GB/s, avg_t=383.84 us
[rank 7] Dispatch bandwidth: 35.31 GB/s, avg_t=212.73 us | Combine bandwidth: 36.41 GB/s, avg_t=399.24 us
[rank 8] Dispatch bandwidth: 34.06 GB/s, avg_t=220.55 us | Combine bandwidth: 36.88 GB/s, avg_t=394.19 us
[rank 9] Dispatch bandwidth: 32.60 GB/s, avg_t=230.44 us | Combine bandwidth: 37.83 GB/s, avg_t=384.23 us
[rank 10] Dispatch bandwidth: 31.97 GB/s, avg_t=234.98 us | Combine bandwidth: 37.60 GB/s, avg_t=386.58 us
[rank 11] Dispatch bandwidth: 30.79 GB/s, avg_t=243.97 us | Combine bandwidth: 38.79 GB/s, avg_t=374.77 us
[rank 12] Dispatch bandwidth: 32.91 GB/s, avg_t=228.23 us | Combine bandwidth: 37.30 GB/s, avg_t=389.72 us
[rank 13] Dispatch bandwidth: 32.08 GB/s, avg_t=234.14 us | Combine bandwidth: 38.24 GB/s, avg_t=380.15 us
[rank 14] Dispatch bandwidth: 33.88 GB/s, avg_t=221.69 us | Combine bandwidth: 37.53 GB/s, avg_t=387.35 us
[rank 15] Dispatch bandwidth: 34.94 GB/s, avg_t=215.00 us | Combine bandwidth: 36.81 GB/s, avg_t=394.93 us


[rank 0] Dispatch send/recv time: 34.99 us | Combine send/recv time: 44.52 us
[rank 1] Dispatch send/recv time: 34.97 us | Combine send/recv time: 44.00 us
[rank 2] Dispatch send/recv time: 34.20 us | Combine send/recv time: 43.26 us
[rank 3] Dispatch send/recv time: 35.05 us | Combine send/recv time: 44.12 us
[rank 4] Dispatch send/recv time: 34.69 us | Combine send/recv time: 43.82 us
[rank 5] Dispatch send/recv time: 35.34 us | Combine send/recv time: 44.74 us
[rank 6] Dispatch send/recv time: 34.73 us | Combine send/recv time: 43.23 us
[rank 7] Dispatch send/recv time: 35.26 us | Combine send/recv time: 44.88 us
[rank 8] Dispatch send/recv time: 34.82 us | Combine send/recv time: 42.72 us
[rank 9] Dispatch send/recv time: 34.51 us | Combine send/recv time: 44.48 us
[rank 10] Dispatch send/recv time: 35.01 us | Combine send/recv time: 42.73 us
[rank 11] Dispatch send/recv time: 34.95 us | Combine send/recv time: 44.91 us
[rank 12] Dispatch send/recv time: 35.43 us | Combine send/recv time: 44.33 us
[rank 13] Dispatch send/recv time: 34.93 us | Combine send/recv time: 43.20 us
[rank 14] Dispatch send/recv time: 34.88 us | Combine send/recv time: 42.51 us
[rank 15] Dispatch send/recv time: 35.04 us | Combine send/recv time: 43.87 us







```





```


