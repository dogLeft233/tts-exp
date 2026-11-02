"""PyTorch SDPA preserving the upstream FlashAttention length semantics.

Upstream's plain SDPA fallback explicitly drops length masks. This adapter is
limited to the noncausal full attention used in the frozen single-GPU protocol.
"""
def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
              softmax_scale=None, q_scale=None, causal=False,
              window_size=(-1,-1), deterministic=False, dtype=None,
              fa_version=None, ip_mask=None):
    import torch
    assert not causal and tuple(window_size)==(-1,-1)
    assert ip_mask is None
    assert q.shape[2] == k.shape[2] == v.shape[2]
    if q_scale is not None:
        q = q*q_scale
    q, k = q.to(v.dtype), k.to(v.dtype)
    mask = None
    if k_lens is not None:
        lengths = torch.as_tensor(k_lens,device=q.device)
        assert torch.all(lengths>0) and torch.all(lengths<=k.shape[1])
        # Broadcast only the key axis; no Lq*Lk mask allocation.
        if not bool(torch.all(lengths==k.shape[1])):
            mask = (torch.arange(k.shape[1],device=q.device)[None,:]<lengths[:,None])[:,None,None,:]
    result = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
        attn_mask=mask,dropout_p=dropout_p,scale=softmax_scale).transpose(1,2).contiguous()
    if q_lens is not None:
        lengths = torch.as_tensor(q_lens,device=q.device)
        result = result.masked_fill(torch.arange(q.shape[1],device=q.device)[None,:,None,None]>=lengths[:,None,None,None],0.)
    return result


def install():
    import src.wan_transformer3d_audio_2512 as flash
    import src.wan_transformer3d_audio as preview
    import src.wan_image_encoder as image
    for module in [flash,preview]:
        module.attention = attention
        module.audio_attention = attention
        module.audio_mask_attention = attention
    image.attention = attention
