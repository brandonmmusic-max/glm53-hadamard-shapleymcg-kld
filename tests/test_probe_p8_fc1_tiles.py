import pytest

from glm53_nvfp4.probe_p8_fc1_tiles import decide


def cell(**kw):
    return dict(matches_control=True,deterministic=True,graph_matches_eager=True,finite=True,**kw)


def test_requires_correctness_before_speed():
    values={'128':[1,1], '64':[.6,.6], '32':[.7,.7]}
    result=decide([cell()],values,.35)
    assert result['speed_pass_by_tile']=={'64':True,'32':False}
    bad=cell()
    bad['matches_control']=False
    assert not any(decide([bad],values,.35)['speed_pass_by_tile'].values())


@pytest.mark.parametrize('values',[
    {'128':[1],'64':[.5]},
    {'128':[1],'64':[],'32':[.5]},
    {'128':[0],'64':[.5],'32':[.5]},
    {'128':[1],'64':[float('nan')],'32':[.5]},
])
def test_invalid_samples_fail_closed(values):
    with pytest.raises(ValueError):
        decide([cell()],values,.35)
