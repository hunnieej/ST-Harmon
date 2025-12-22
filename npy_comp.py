import numpy as np

a = np.load("importance15.npy", allow_pickle=True)
b = np.load("importance16.npy", allow_pickle=True)
if a.shape != b.shape:
    print("Different shape:", a.shape, b.shape)
else:
    diff = a - b
    # float/정수 모두에서 유용한 지표들
    print("max abs diff:", np.max(np.abs(diff)))
    print("num different (exact):", np.count_nonzero(a != b))

    # float일 때 allclose 기준으로 다른 위치 인덱스 보고 싶으면
    mask = ~np.isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=True)
    print("num different (isclose):", np.count_nonzero(mask))
    if np.any(mask):
        idx = np.argwhere(mask)[0]
        print("first mismatch index:", tuple(idx))
        print("a[idx], b[idx]:", a[tuple(idx)], b[tuple(idx)])
