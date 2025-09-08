import numpy as np

def _is_pos_def(x):
    """Kiểm tra xem ma trận có xác định dương không."""
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.LinAlgError:
        return False

def nearest_positive_definite(a):
    """Trả về ma trận xác định dương gần nhất với ma trận đầu vào (Higham 1988).
    Kiểm tra và trả về cờ nếu diagonal elements > 1.0."""
    b = (a + a.T) / 2
    _, s, vh = np.linalg.svd(b)
    h = vh.T @ np.diag(s) @ vh
    a2 = (b + h) / 2
    a3 = (a2 + a2.T) / 2
    
    diag_gt_one = False
    if np.any(np.diag(a3) > 1.0 + np.finfo(float).eps): 
        diag_gt_one = True

    if _is_pos_def(a3):
        return a3, diag_gt_one
    
    spacing = np.spacing(np.linalg.norm(a))
    identity = np.eye(a.shape[0])
    k = 1
    while not _is_pos_def(a3):
        min_eig = np.min(np.real(np.linalg.eigvals(a3)))
        a3 += identity * (-min_eig * k**2 + spacing)
        k += 1
        if k > 1000:
            raise RuntimeError("Không thể tìm thấy ma trận xác định dương.")
    if np.any(np.diag(a3) > 1.0 + np.finfo(float).eps):
        diag_gt_one = True
    return a3, diag_gt_one
