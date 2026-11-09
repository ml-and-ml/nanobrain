import numpy as np
import nibabel as nib
import pytest
import torch

from nanobrain.model import ViTMAE3D


def tiny(mode):
    return ViTMAE3D(grid_size=(16,16,16),patch_size=4,depth=1,embed_dim=32,
                    num_heads=4,decoder_depth=1,decoder_embed_dim=32,
                    decoder_num_heads=4,class_tokens=0,region_mode=mode)


@pytest.mark.parametrize('mode',['regions','consistency','coverage'])
def test_backbone_initialization_preserved(mode):
    torch.manual_seed(7338)
    baseline=tiny('none')
    original_rng=torch.get_rng_state()
    torch.manual_seed(7338)
    modified=tiny(mode)
    assert torch.equal(original_rng,torch.get_rng_state())
    for name,value in baseline.state_dict().items():
        torch.testing.assert_close(value,modified.state_dict()[name],rtol=0,atol=0)


@pytest.mark.parametrize('mode',['regions','consistency','coverage'])
def test_training_losses_and_gradient_path(mode):
    torch.manual_seed(11)
    model=tiny(mode)
    images=torch.randn(4,16,16,16)
    mask=torch.ones_like(images)
    weights=torch.zeros(4,64,21)
    weights[:,:,0]=0.5
    weights[:,::2,8]=0.5
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss,state=model(images,mask,16,32,2,region_weights=weights)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert model.patch_embed.weight.grad.abs().sum()>0
    if mode=='regions':
        assert model.region_context.region_identity.grad.abs().sum()>0
    if mode=='consistency':
        assert state['region_consistency_loss']>0
    for visible,target in zip(state['vis_ids'],state['target_ids']):
        assert not set(visible.tolist()).intersection(target.tolist())


def test_real_atlas_contextual_output_and_checkpoint(tmp_path):
    model=tiny('regions').eval()
    affine=np.eye(4)
    affine[:3,3]=[-35,-32,-21]
    rng=np.random.default_rng(3)
    image=nib.Nifti1Image(rng.normal(size=(16,16,16)).astype(np.float32),affine)
    mask=nib.Nifti1Image(np.ones((16,16,16),dtype=np.uint8),affine)
    sample=model.transform({'image':image,'brain_mask':mask,'mni_affine':np.eye(4)})
    transformed=model.transform_cuda(sample)
    prior=model.sample_region_weights(transformed)
    assert prior[:,:,8].sum()>0  # Actual left hippocampus, no guessed spatial bin.
    named=model.forward_region_embeddings([sample])[0]
    assert named['names'][8]=='left_hippocampus'
    assert named['available'][8]
    assert named['embeddings'].shape==(21,32)
    standard=model.forward_embeddings([sample],return_dense=True)[0]
    torch.testing.assert_close(named['global_embedding'],standard['global_embedding'])
    assert standard['dense_embedding'].shape==(4,4,4,32)
    checkpoint=tmp_path/'state.pt'
    torch.save(model.state_dict(),checkpoint)
    restored=tiny('regions').eval()
    restored.load_state_dict(torch.load(checkpoint,weights_only=True))
    torch.testing.assert_close(restored.forward_region_embeddings([sample])[0]['embeddings'],named['embeddings'])
    with pytest.raises(ValueError,match='MNI affine'):
        model.transform({'image':image,'brain_mask':mask})
