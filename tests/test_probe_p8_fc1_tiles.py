import pytest

from glm53_nvfp4.probe_p8_fc1_tiles import decide, canonical_rows


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


def metadata():
    counts=[2]+[0]*286+[1]
    bases=[0]+[1]*287+[2]
    mapping=[0]*32
    mapping[0],mapping[1],mapping[16]=1,0,2
    return counts,bases,mapping


def test_canonicalization_preserves_logical_inputs_despite_row_permutation():
    counts,bases,mapping=metadata()
    assert canonical_rows(counts,bases,mapping,3)==[1,0,16]
    mapping[0],mapping[1]=0,1
    assert canonical_rows(counts,bases,mapping,3)==[0,1,16]


@pytest.mark.parametrize('fault',['duplicate','prefix','terminal','missing'])
def test_invalid_canonical_mapping_fails(fault):
    counts,bases,mapping=metadata()
    if fault=='duplicate': mapping[1]=1
    if fault=='prefix': bases[3]=2
    if fault=='terminal': bases[-1]=3
    if fault=='missing': counts[-1]=0
    with pytest.raises(ValueError):
        canonical_rows(counts,bases,mapping,3)
