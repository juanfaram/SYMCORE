from online_phase2_temporal import stream,slope
def test_phase2_stream_has_many_checkpoints_and_no_batch_dependency():
 rows=stream(0);assert len(rows)>=20 and all(rows[i][0]<rows[i+1][0] for i in range(len(rows)-1))
def test_known_instrument_has_expected_global_signs():
 rows=stream(1);assert slope(rows,1)>0 and slope(rows,2)<0
def test_late_learning_can_saturate_without_reversing():
 rows=stream(2);early=[x for x in rows if x[0]<=500];late=[x for x in rows if x[0]>=550]
 assert slope(early,1)>slope(late,1)>0
